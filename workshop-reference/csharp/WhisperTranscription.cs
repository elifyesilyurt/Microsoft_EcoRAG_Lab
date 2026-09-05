using Microsoft.AI.Foundry.Local;
using Microsoft.Extensions.Logging.Abstractions;

namespace Examples;

/// <summary>
/// Part 9: Whisper voice transcription with Foundry Local.
/// Transcribes WAV audio files using the OpenAI Whisper model running locally.
/// Uses the Foundry Local SDK's built-in audio client.
/// </summary>
public static class WhisperTranscription
{
    public static async Task RunAsync(string[] args)
    {
        var alias = "whisper-medium";
        var samplesDir = FindSamplesDirectory();

        // Determine which files to transcribe
        string[] audioFiles;
        if (args.Length > 1)
        {
            // A specific file was passed after the "whisper" command
            var filePath = args[1];
            if (!File.Exists(filePath))
            {
                Console.WriteLine($"Audio file not found: {filePath}");
                Console.WriteLine("Usage: dotnet run whisper <path-to-wav-file>");
                return;
            }
            audioFiles = [filePath];
        }
        else
        {
            // Default: transcribe all Zava sample WAV files
            if (!Directory.Exists(samplesDir))
            {
                Console.WriteLine($"Samples directory not found: {samplesDir}");
                Console.WriteLine("Run 'python samples/audio/generate_samples.py' first.");
                return;
            }
            audioFiles = Directory.GetFiles(samplesDir, "zava-*.wav")
                .OrderBy(f => f).ToArray();

            if (audioFiles.Length == 0)
            {
                Console.WriteLine($"No WAV files found in {samplesDir}");
                Console.WriteLine("Run 'python samples/audio/generate_samples.py' first.");
                return;
            }
        }

        // Step 1: Start the Foundry Local service
        Console.WriteLine("Starting Foundry Local service...");
        await FoundryLocalManager.CreateAsync(
            new Configuration
            {
                AppName = "FoundryLocalSamples",
                Web = new Configuration.WebService { Urls = "http://127.0.0.1:0" }
            }, NullLogger.Instance, default);
        var manager = FoundryLocalManager.Instance;
        await manager.StartWebServiceAsync(default);

        // Step 2: Get the model from the catalog
        var catalog = await manager.GetCatalogAsync(default);
        var model = await catalog.GetModelAsync(alias, default);

        // Step 3: Check if the model is already downloaded
        var isCached = await model.IsCachedAsync(default);

        if (isCached)
        {
            Console.WriteLine($"Model already downloaded: {alias}");
        }
        else
        {
            Console.WriteLine($"Downloading model: {alias} (this may take several minutes)...");
            await model.DownloadAsync(null, default);
            Console.WriteLine($"Download complete: {alias}");
        }

        // Step 4: Load the model into memory
        Console.WriteLine($"Loading model: {alias}...");
        await model.LoadAsync(default);
        Console.WriteLine($"Loaded model: {model.Id}\n");

        // Step 5: Get the audio client from the Foundry Local SDK
        var audioClient = await model.GetAudioClientAsync();
        audioClient.Settings.Language = "en";

        // Step 6: Transcribe each audio file
        foreach (var audioPath in audioFiles)
        {
            var filename = Path.GetFileName(audioPath);
            Console.WriteLine(new string('=', 60));
            Console.WriteLine($"File: {filename}");
            Console.WriteLine(new string('=', 60));

            // Split long audio into 30-second chunks to work around Whisper's
            // 30-second encoder window. See https://github.com/microsoft/Foundry-Local/issues/517
            var chunkPaths = SplitWavIntoChunks(audioPath);

            if (chunkPaths == null)
            {
                // Short audio (≤30 s) — single pass
                var result = await audioClient.TranscribeAudioAsync(audioPath);
                Console.WriteLine(result.Text);
            }
            else
            {
                // Long audio — transcribe each chunk and concatenate
                var parts = new List<string>();
                for (var i = 0; i < chunkPaths.Count; i++)
                {
                    Console.Write($"  Chunk {i + 1}/{chunkPaths.Count}... ");
                    var result = await audioClient.TranscribeAudioAsync(chunkPaths[i]);
                    Console.WriteLine("done");
                    if (!string.IsNullOrWhiteSpace(result.Text))
                        parts.Add(result.Text);
                }
                Console.WriteLine(string.Join(" ", parts));

                // Clean up temp files
                var tmpDir = Path.GetDirectoryName(chunkPaths[0])!;
                foreach (var p in chunkPaths) File.Delete(p);
                Directory.Delete(tmpDir);
            }
            Console.WriteLine();
        }

        Console.WriteLine($"Done — transcribed {audioFiles.Length} file(s).");

        // Cleanup: unload the model to release resources
        await model.UnloadAsync(default);
    }

    /// <summary>
    /// Walk up from both AppContext.BaseDirectory and the current directory
    /// to locate the samples/audio folder, handling varying build output depths.
    /// </summary>
    private static string FindSamplesDirectory()
    {
        var candidates = new[] { AppContext.BaseDirectory, Directory.GetCurrentDirectory() };
        foreach (var start in candidates)
        {
            var dir = start;
            for (var i = 0; i < 8; i++)
            {
                var samplesPath = Path.Combine(dir, "samples", "audio");
                if (Directory.Exists(samplesPath))
                    return samplesPath;

                var parent = Directory.GetParent(dir);
                if (parent == null) break;
                dir = parent.FullName;
            }
        }
        return Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "samples", "audio");
    }

    /// <summary>
    /// Split a WAV file into ≤30-second chunk files. Returns null if the file
    /// is already 30 seconds or shorter. Temporary files are written to a
    /// unique temp directory and the caller is responsible for cleanup.
    /// </summary>
    private static List<string>? SplitWavIntoChunks(string wavPath)
    {
        const int chunkSeconds = 30;
        const int headerSize = 44;

        var fileBytes = File.ReadAllBytes(wavPath);
        var numChannels = BitConverter.ToUInt16(fileBytes, 22);
        var sampleRate = BitConverter.ToInt32(fileBytes, 24);
        var bitsPerSample = BitConverter.ToUInt16(fileBytes, 34);
        var bytesPerSample = bitsPerSample / 8 * numChannels;

        var pcmData = fileBytes.AsSpan(headerSize);
        var totalSamples = pcmData.Length / bytesPerSample;
        var chunkSamples = chunkSeconds * sampleRate;

        if (totalSamples <= chunkSamples) return null;

        var numChunks = (totalSamples + chunkSamples - 1) / chunkSamples;
        var duration = (double)totalSamples / sampleRate;
        Console.WriteLine($"  Audio is {duration:F1}s — splitting into {numChunks} chunks of {chunkSeconds}s");

        var tmpDir = Path.Combine(Path.GetTempPath(), $"whisper-chunks-{Guid.NewGuid():N}");
        Directory.CreateDirectory(tmpDir);
        var chunkPaths = new List<string>();

        for (var i = 0; i < numChunks; i++)
        {
            var startByte = i * chunkSamples * bytesPerSample;
            var endByte = Math.Min(startByte + chunkSamples * bytesPerSample, pcmData.Length);
            var chunkLen = endByte - startByte;

            var header = new byte[headerSize];
            System.Text.Encoding.ASCII.GetBytes("RIFF").CopyTo(header, 0);
            BitConverter.GetBytes(36 + chunkLen).CopyTo(header, 4);
            System.Text.Encoding.ASCII.GetBytes("WAVE").CopyTo(header, 8);
            System.Text.Encoding.ASCII.GetBytes("fmt ").CopyTo(header, 12);
            BitConverter.GetBytes(16).CopyTo(header, 16);          // fmt chunk size
            BitConverter.GetBytes((ushort)1).CopyTo(header, 20);   // PCM
            BitConverter.GetBytes(numChannels).CopyTo(header, 22);
            BitConverter.GetBytes(sampleRate).CopyTo(header, 24);
            BitConverter.GetBytes(sampleRate * bytesPerSample).CopyTo(header, 28);
            BitConverter.GetBytes((ushort)bytesPerSample).CopyTo(header, 32);
            BitConverter.GetBytes(bitsPerSample).CopyTo(header, 34);
            System.Text.Encoding.ASCII.GetBytes("data").CopyTo(header, 36);
            BitConverter.GetBytes(chunkLen).CopyTo(header, 40);

            var chunkPath = Path.Combine(tmpDir, $"chunk-{i}.wav");
            using var fs = File.Create(chunkPath);
            fs.Write(header);
            fs.Write(pcmData.Slice(startByte, chunkLen));
            chunkPaths.Add(chunkPath);
        }

        return chunkPaths;
    }
}