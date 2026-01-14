import httpx
import asyncio

class ChatterboxTTS:
    """Edge-TTS style client for Chatterbox TTS Server."""

    def __init__(self, base_url: str = "http://172.16.6.19:8000"):
        self.base_url = base_url

    async def generate(
        self,
        text: str,
        voice: str = None,
        output_file: str = "output.wav"
    ) -> bytes:
        """Generate speech and save to file."""
        params = {"text": text}
        if voice:
            params["voice"] = voice

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.get(f"{self.base_url}/tts", params=params)
            response.raise_for_status()

            audio_data = response.content
            with open(output_file, "wb") as f:
                f.write(audio_data)

            return audio_data

    def generate_sync(
        self,
        text: str,
        voice: str = None,
        output_file: str = "output.wav"
    ) -> bytes:
        """Synchronous version."""
        return asyncio.run(self.generate(text, voice, output_file))


# Usage example (like edge-tts)
async def main():
    tts = ChatterboxTTS()

    # Simple usage
    await tts.generate(
        text="Namaste! is a test of the Chatterbox TTS server.",
        output_file="test_output_3.wav"
    )
    print("Audio saved to test_output.wav")


if __name__ == "__main__":
    asyncio.run(main())
