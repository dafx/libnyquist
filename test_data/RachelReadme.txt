Raw multitrack from Anna Blanton's 'Rachel'. This file is provided for educational purposes only, and the material contained in it should not be used for any commercial purpose without the express permission of the copyright holders. Please refer to https://creativecommons.org/licenses/by-sa/4.0/ and www.cambridge-mt.com for further licensing details.

Comprises 10 WAV files at 24-bit/44.1kHz resolution.

Tempo: approx 63.5bpm.

-------------------

File download from https://cambridge-mt.com/ms/mtk/#AnnaBlanton

```
sox --combine merge 01_Congas.wav 02_Bass1.wav 04_UkeleleMic.wav 06_Violin.wav 07_Viola.wav 08_Cello.wav 09_LeadVox.wav 10_BackingVox.wav output8ch.wav

ffmpeg -i output8ch.wav -c:a libopus Rachel8ch.opus
```