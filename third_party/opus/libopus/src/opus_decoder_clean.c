/* Copyright (c) 2010 Xiph.Org Foundation, Skype Limited
   Written by Jean-Marc Valin and Koen Vos */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifndef OPUS_BUILD
#error "OPUS_BUILD _MUST_ be defined to build Opus. This probably means you need other defines as well, as in a config.h. See the included build files for details."
#endif

#if defined(__GNUC__) && (__GNUC__ >= 2) && !defined(__OPTIMIZE__)
#pragma message "You appear to be compiling without optimization, if so opus will be very slow."
#endif

#include <stdarg.h>
#include "celt.h"
#include "opus.h"
#include "entdec.h"
#include "modes.h"
#include "API.h"
#include "stack_alloc.h"
#include "float_cast.h"
#include "opus_private.h"
#include "os_support.h"
#include "structs.h"
#include "define.h"
#include "mathops.h"
#include "cpu_support.h"

struct OpusDecoder
{
    int celt_dec_offset;
    int silk_dec_offset;
    int channels;
    opus_int32 Fs; /** Sampling rate (at the API level) */
    silk_DecControlStruct DecControl;
    int decode_gain;

    /* Everything beyond this point gets cleared on a reset */
#define OPUS_DECODER_RESET_START stream_channels
    int stream_channels;

    int bandwidth;
    int mode;
    int prev_mode;
    int frame_size;
    int prev_redundancy;
    int last_packet_duration;
#ifndef FIXED_POINT
    opus_val16 softclip_mem[2];
#endif

    opus_uint32 rangeFinal;
};

#ifdef FIXED_POINT
static OPUS_INLINE opus_int16 SAT16(opus_int32 x)
{
    return x > 32767 ? 32767 : x < -32768 ? -32768
                                          : (opus_int16)x;
}
#endif

int opus_decoder_get_size(int channels)
{
    int silkDecSizeBytes, celtDecSizeBytes;
    int ret;
    if (channels < 1 || channels > 2)
        return 0;
    ret = silk_Get_Decoder_Size(&silkDecSizeBytes);
    if (ret)
        return 0;
    silkDecSizeBytes = align(silkDecSizeBytes);
    celtDecSizeBytes = celt_decoder_get_size(channels);
    return align(sizeof(OpusDecoder)) + silkDecSizeBytes + celtDecSizeBytes;
}

int opus_decoder_init(OpusDecoder *st, opus_int32 Fs, int channels)
{
    void *silk_dec;
    CELTDecoder *celt_dec;
    int ret, silkDecSizeBytes;

    if ((Fs != 48000 && Fs != 24000 && Fs != 16000 && Fs != 12000 && Fs != 8000) || (channels != 1 && channels != 2))
        return OPUS_BAD_ARG;

    OPUS_CLEAR((char *)st, opus_decoder_get_size(channels));
    /* Initialize SILK encoder */
    ret = silk_Get_Decoder_Size(&silkDecSizeBytes);
    if (ret)
        return OPUS_INTERNAL_ERROR;

    silkDecSizeBytes = align(silkDecSizeBytes);
    st->silk_dec_offset = align(sizeof(OpusDecoder));
    st->celt_dec_offset = st->silk_dec_offset + silkDecSizeBytes;
    silk_dec = (char *)st + st->silk_dec_offset;
    celt_dec = (CELTDecoder *)((char *)st + st->celt_dec_offset);
    st->stream_channels = st->channels = channels;

    st->Fs = Fs;
    st->DecControl.API_sampleRate = st->Fs;
    st->DecControl.nChannelsAPI = st->channels;

    /* Reset decoder */
    ret = silk_InitDecoder(silk_dec);
    if (ret)
        return OPUS_INTERNAL_ERROR;

    /* Initialize CELT decoder */
    ret = celt_decoder_init(celt_dec, Fs, channels);
    if (ret != OPUS_OK)
        return OPUS_INTERNAL_ERROR;

    celt_decoder_ctl(celt_dec, CELT_SET_SIGNALLING(0));

    st->prev_mode = 0;
    st->frame_size = Fs / 400;
    return OPUS_OK;
}

OpusDecoder *opus_decoder_create(opus_int32 Fs, int channels, int *error)
{
    int ret;
    OpusDecoder *st;
    if ((Fs != 48000 && Fs != 24000 && Fs != 16000 && Fs != 12000 && Fs != 8000) || (channels != 1 && channels != 2))
    {
        if (error)
            *error = OPUS_BAD_ARG;
        return NULL;
    }
    st = (OpusDecoder *)opus_alloc(opus_decoder_get_size(channels));
    if (st == NULL)
    {
        if (error)
            *error = OPUS_ALLOC_FAIL;
        return NULL;
    }
    ret = opus_decoder_init(st, Fs, channels);
    if (error)
        *error = ret;
    if (ret != OPUS_OK)
    {
        opus_free(st);
        st = NULL;
    }
    return st;
}

static void smooth_fade(const opus_val16 *in1, const opus_val16 *in2,
                        opus_val16 *out, int overlap, int channels,
                        const opus_val16 *window, opus_int32 Fs)
{
    int i, c;
    int inc = 48000 / Fs;
    for (c = 0; c < channels; c++)
    {
        for (i = 0; i < overlap; i++)
        {
            opus_val16 w = MULT16_16_Q15(window[i * inc], window[i * inc]);
            out[i * channels + c] = SHR32(MAC16_16(MULT16_16(w, in2[i * channels + c]),
                                                   Q15ONE - w, in1[i * channels + c]),
                                          15);
        }
    }
}

static int opus_packet_get_mode(const unsigned char *data)
{
    int mode;
    if (data[0] & 0x80)
    {
        mode = MODE_CELT_ONLY;
    }
    else if ((data[0] & 0x60) == 0x60)
    {
        mode = MODE_HYBRID;
    }
    else
    {
        mode = MODE_SILK_ONLY;
    }
    return mode;
}

static int opus_decode_frame(OpusDecoder *st, const unsigned char *data,
                             opus_int32 len, opus_val16 *pcm, int frame_size, int decode_fec)
{
    CELTDecoder *celt_dec;
    int i, celt_ret = 0;
    ec_dec dec;
    int redundant_audio_size;
    VARDECL(opus_val16, redundant_audio);

    int audiosize;
    int mode;
    int start_band;
    int c;
    int F2_5, F5, F10, F20;
    const opus_val16 *window;
    ALLOC_STACK;

    celt_dec = (CELTDecoder *)((char *)st + st->celt_dec_offset);
    F20 = st->Fs / 50;
    F10 = F20 >> 1;
    F5 = F10 >> 1;
    F2_5 = F5 >> 1;
    if (frame_size < F2_5)
    {
        RESTORE_STACK;
        return OPUS_BUFFER_TOO_SMALL;
    }
    /* Limit frame_size to avoid excessive stack allocations. */
    frame_size = IMIN(frame_size, st->Fs / 25 * 3);
    /* Payloads of 1 (2 including ToC) or 0 trigger the PLC/DTX */
    if (len <= 1)
    {
        data = NULL;
        /* In that case, don't conceal more than what the ToC says */
        frame_size = IMIN(frame_size, st->frame_size);
    }
    if (data != NULL)
    {
        audiosize = st->frame_size;
        mode = st->mode;
        ec_dec_init(&dec, (unsigned char *)data, len);
    }
    else
    {
        audiosize = 0;
        celt_assert(0);
    }
    if (audiosize > frame_size)
    {
        /*fprintf(stderr, "PCM buffer too small: %d vs %d (mode = %d)\n", audiosize, frame_size, mode);*/
        RESTORE_STACK;
        return OPUS_BAD_ARG;
    }
    else
    {
        frame_size = audiosize;
    }

    /* SILK processing */
    if (mode != MODE_CELT_ONLY)
    {
        celt_assert(0);
    }

    start_band = 0;

    {
        int endband = 21;

        switch (st->bandwidth)
        {
        case OPUS_BANDWIDTH_NARROWBAND:
            endband = 13;
            break;
        case OPUS_BANDWIDTH_MEDIUMBAND:
        case OPUS_BANDWIDTH_WIDEBAND:
            endband = 17;
            break;
        case OPUS_BANDWIDTH_SUPERWIDEBAND:
            endband = 19;
            break;
        case OPUS_BANDWIDTH_FULLBAND:
            endband = 21;
            break;
        }
        celt_decoder_ctl(celt_dec, CELT_SET_END_BAND(endband));
        celt_decoder_ctl(celt_dec, CELT_SET_CHANNELS(st->stream_channels));
    }

    /* MUST be after PLC */
    celt_decoder_ctl(celt_dec, CELT_SET_START_BAND(start_band));

    if (mode != MODE_SILK_ONLY)
    {
        int celt_frame_size = IMIN(F20, frame_size);
        /* Make sure to discard any previous CELT state */
        if (mode != st->prev_mode && st->prev_mode > 0 && !st->prev_redundancy)
            celt_decoder_ctl(celt_dec, OPUS_RESET_STATE);
        /* Decode CELT */
        celt_ret = celt_decode_with_ec(celt_dec, decode_fec ? NULL : data,
                                       len, pcm, celt_frame_size, &dec);
    }
    else
    {
        celt_assert(0);
    }

    if (mode != MODE_CELT_ONLY)
    {
        celt_assert(0);
    }

    {
        const CELTMode *celt_mode;
        celt_decoder_ctl(celt_dec, CELT_GET_MODE(&celt_mode));
        window = celt_mode->window;
    }

    if (st->decode_gain)
    {
        opus_val32 gain;
        gain = celt_exp2(MULT16_16_P15(QCONST16(6.48814081e-4f, 25), st->decode_gain));
        for (i = 0; i < frame_size * st->channels; i++)
        {
            opus_val32 x;
            x = MULT16_32_P16(pcm[i], gain);
            pcm[i] = SATURATE(x, 32767);
        }
    }

    if (len <= 1)
        st->rangeFinal = 0;
    else
        st->rangeFinal = dec.rng ^ 0;

    st->prev_mode = mode;
    st->prev_redundancy = 0;

    if (celt_ret >= 0)
    {
        if (OPUS_CHECK_ARRAY(pcm, audiosize * st->channels))
            OPUS_PRINT_INT(audiosize);
    }

    RESTORE_STACK;
    return celt_ret < 0 ? celt_ret : audiosize;
}

int opus_decode_native(OpusDecoder *st, const unsigned char *data,
                       opus_int32 len, opus_val16 *pcm, int frame_size, int decode_fec,
                       int self_delimited, opus_int32 *packet_offset, int soft_clip)
{
    int i, nb_samples;
    int count, offset;
    unsigned char toc;
    int packet_frame_size, packet_bandwidth, packet_mode, packet_stream_channels;
    /* 48 x 2.5 ms = 120 ms */
    opus_int16 size[48];
    if (decode_fec < 0 || decode_fec > 1)
        return OPUS_BAD_ARG;
    /* For FEC/PLC, frame_size has to be to have a multiple of 2.5 ms */
    if ((decode_fec || len == 0 || data == NULL) && frame_size % (st->Fs / 400) != 0)
        return OPUS_BAD_ARG;

    packet_mode = opus_packet_get_mode(data);
    packet_bandwidth = opus_packet_get_bandwidth(data);
    packet_frame_size = opus_packet_get_samples_per_frame(data, st->Fs);
    packet_stream_channels = opus_packet_get_nb_channels(data);

    count = opus_packet_parse_impl(data, len, self_delimited, &toc, NULL,
                                   size, &offset, packet_offset);
    if (count < 0)
        return count;

    data += offset;

    if (count * packet_frame_size > frame_size)
        return OPUS_BUFFER_TOO_SMALL;

    /* Update the state as the last step to avoid updating it on an invalid packet */
    st->mode = packet_mode;
    st->bandwidth = packet_bandwidth;
    st->frame_size = packet_frame_size;
    st->stream_channels = packet_stream_channels;

    nb_samples = 0;
    for (i = 0; i < count; i++)
    {
        int ret;
        ret = opus_decode_frame(st, data, size[i], pcm + nb_samples * st->channels, frame_size - nb_samples, 0);
        if (ret < 0)
            return ret;
        celt_assert(ret == packet_frame_size);
        data += size[i];
        nb_samples += ret;
    }
    st->last_packet_duration = nb_samples;
    st->softclip_mem[0] = st->softclip_mem[1] = 0;
    return nb_samples;
}

#ifdef FIXED_POINT

int opus_decode(OpusDecoder *st, const unsigned char *data,
                opus_int32 len, opus_val16 *pcm, int frame_size, int decode_fec)
{
    if (frame_size <= 0)
        return OPUS_BAD_ARG;
    return opus_decode_native(st, data, len, pcm, frame_size, decode_fec, 0, NULL, 0);
}

#ifndef DISABLE_FLOAT_API
int opus_decode_float(OpusDecoder *st, const unsigned char *data,
                      opus_int32 len, float *pcm, int frame_size, int decode_fec)
{
    VARDECL(opus_int16, out);
    int ret, i;
    ALLOC_STACK;

    if (frame_size <= 0)
    {
        RESTORE_STACK;
        return OPUS_BAD_ARG;
    }
    ALLOC(out, frame_size * st->channels, opus_int16);

    ret = opus_decode_native(st, data, len, out, frame_size, decode_fec, 0, NULL, 0);
    if (ret > 0)
    {
        for (i = 0; i < ret * st->channels; i++)
            pcm[i] = (1.f / 32768.f) * (out[i]);
    }
    RESTORE_STACK;
    return ret;
}
#endif

#else
int opus_decode(OpusDecoder *st, const unsigned char *data,
                opus_int32 len, opus_int16 *pcm, int frame_size, int decode_fec)
{
    VARDECL(float, out);
    int ret, i;
    ALLOC_STACK;

    if (frame_size <= 0)
    {
        RESTORE_STACK;
        return OPUS_BAD_ARG;
    }

    ALLOC(out, frame_size * st->channels, float);

    ret = opus_decode_native(st, data, len, out, frame_size, decode_fec, 0, NULL, 1);
    if (ret > 0)
    {
        for (i = 0; i < ret * st->channels; i++)
            pcm[i] = FLOAT2INT16(out[i]);
    }
    RESTORE_STACK;
    return ret;
}

int opus_decode_float(OpusDecoder *st, const unsigned char *data,
                      opus_int32 len, opus_val16 *pcm, int frame_size, int decode_fec)
{
    if (frame_size <= 0)
        return OPUS_BAD_ARG;
    return opus_decode_native(st, data, len, pcm, frame_size, decode_fec, 0, NULL, 0);
}

#endif

int opus_decoder_ctl(OpusDecoder *st, int request, ...)
{
    int ret = OPUS_OK;
    va_list ap;
    void *silk_dec;
    CELTDecoder *celt_dec;

    silk_dec = (char *)st + st->silk_dec_offset;
    celt_dec = (CELTDecoder *)((char *)st + st->celt_dec_offset);

    va_start(ap, request);

    switch (request)
    {
    case OPUS_GET_BANDWIDTH_REQUEST:
    {
        opus_int32 *value = va_arg(ap, opus_int32 *);
        if (!value)
        {
            goto bad_arg;
        }
        *value = st->bandwidth;
    }
    break;
    case OPUS_GET_FINAL_RANGE_REQUEST:
    {
        opus_uint32 *value = va_arg(ap, opus_uint32 *);
        if (!value)
        {
            goto bad_arg;
        }
        *value = st->rangeFinal;
    }
    break;
    case OPUS_RESET_STATE:
    {
        OPUS_CLEAR((char *)&st->OPUS_DECODER_RESET_START,
                   sizeof(OpusDecoder) -
                       ((char *)&st->OPUS_DECODER_RESET_START - (char *)st));

        celt_decoder_ctl(celt_dec, OPUS_RESET_STATE);
        silk_InitDecoder(silk_dec);
        st->stream_channels = st->channels;
        st->frame_size = st->Fs / 400;
    }
    break;
    case OPUS_GET_SAMPLE_RATE_REQUEST:
    {
        opus_int32 *value = va_arg(ap, opus_int32 *);
        if (!value)
        {
            goto bad_arg;
        }
        *value = st->Fs;
    }
    break;
    case OPUS_GET_PITCH_REQUEST:
    {
        opus_int32 *value = va_arg(ap, opus_int32 *);
        if (!value)
        {
            goto bad_arg;
        }
        if (st->prev_mode == MODE_CELT_ONLY)
            celt_decoder_ctl(celt_dec, OPUS_GET_PITCH(value));
        else
            *value = st->DecControl.prevPitchLag;
    }
    break;
    case OPUS_GET_GAIN_REQUEST:
    {
        opus_int32 *value = va_arg(ap, opus_int32 *);
        if (!value)
        {
            goto bad_arg;
        }
        *value = st->decode_gain;
    }
    break;
    case OPUS_SET_GAIN_REQUEST:
    {
        opus_int32 value = va_arg(ap, opus_int32);
        if (value < -32768 || value > 32767)
        {
            goto bad_arg;
        }
        st->decode_gain = value;
    }
    break;
    case OPUS_GET_LAST_PACKET_DURATION_REQUEST:
    {
        opus_uint32 *value = va_arg(ap, opus_uint32 *);
        if (!value)
        {
            goto bad_arg;
        }
        *value = st->last_packet_duration;
    }
    break;
    default:
        /*fprintf(stderr, "unknown opus_decoder_ctl() request: %d", request);*/
        ret = OPUS_UNIMPLEMENTED;
        break;
    }

    va_end(ap);
    return ret;
bad_arg:
    va_end(ap);
    return OPUS_BAD_ARG;
}

void opus_decoder_destroy(OpusDecoder *st)
{
    opus_free(st);
}

int opus_packet_get_bandwidth(const unsigned char *data)
{
    int bandwidth;
    if (data[0] & 0x80)
    {
        bandwidth = OPUS_BANDWIDTH_MEDIUMBAND + ((data[0] >> 5) & 0x3);
        if (bandwidth == OPUS_BANDWIDTH_MEDIUMBAND)
            bandwidth = OPUS_BANDWIDTH_NARROWBAND;
    }
    else if ((data[0] & 0x60) == 0x60)
    {
        bandwidth = (data[0] & 0x10) ? OPUS_BANDWIDTH_FULLBAND : OPUS_BANDWIDTH_SUPERWIDEBAND;
    }
    else
    {
        bandwidth = OPUS_BANDWIDTH_NARROWBAND + ((data[0] >> 5) & 0x3);
    }
    return bandwidth;
}

int opus_packet_get_samples_per_frame(const unsigned char *data,
                                      opus_int32 Fs)
{
    int audiosize;
    if (data[0] & 0x80)
    {
        audiosize = ((data[0] >> 3) & 0x3);
        audiosize = (Fs << audiosize) / 400;
    }
    else if ((data[0] & 0x60) == 0x60)
    {
        audiosize = (data[0] & 0x08) ? Fs / 50 : Fs / 100;
    }
    else
    {
        audiosize = ((data[0] >> 3) & 0x3);
        if (audiosize == 3)
            audiosize = Fs * 60 / 1000;
        else
            audiosize = (Fs << audiosize) / 100;
    }
    return audiosize;
}

int opus_packet_get_nb_channels(const unsigned char *data)
{
    return (data[0] & 0x4) ? 2 : 1;
}

int opus_packet_get_nb_frames(const unsigned char packet[], opus_int32 len)
{
    int count;
    if (len < 1)
        return OPUS_BAD_ARG;
    count = packet[0] & 0x3;
    if (count == 0)
        return 1;
    else if (count != 3)
        return 2;
    else if (len < 2)
        return OPUS_INVALID_PACKET;
    else
        return packet[1] & 0x3F;
}

int opus_packet_get_nb_samples(const unsigned char packet[], opus_int32 len,
                               opus_int32 Fs)
{
    int samples;
    int count = opus_packet_get_nb_frames(packet, len);

    if (count < 0)
        return count;

    samples = count * opus_packet_get_samples_per_frame(packet, Fs);
    /* Can't have more than 120 ms */
    if (samples * 25 > Fs * 3)
        return OPUS_INVALID_PACKET;
    else
        return samples;
}

int opus_decoder_get_nb_samples(const OpusDecoder *dec,
                                const unsigned char packet[], opus_int32 len)
{
    return opus_packet_get_nb_samples(packet, len, dec->Fs);
}
