/* Copyright (c) 2007-2008 CSIRO
   Copyright (c) 2007-2010 Xiph.Org Foundation
   Copyright (c) 2008 Gregory Maxwell
   Written by Jean-Marc Valin and Gregory Maxwell */
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

#define CELT_DECODER_C

#include "cpu_support.h"
#include "os_support.h"
#include "mdct.h"
#include <math.h>
#include "celt.h"
#include "pitch.h"
#include "bands.h"
#include "modes.h"
#include "entcode.h"
#include "quant_bands.h"
#include "rate.h"
#include "stack_alloc.h"
#include "mathops.h"
#include "float_cast.h"
#include <stdarg.h>
#include "celt_lpc.h"
#include "vq.h"

/**********************************************************************/
/*                                                                    */
/*                             DECODER                                */
/*                                                                    */
/**********************************************************************/
#define DECODE_BUFFER_SIZE 2048

/** Decoder state
 @brief Decoder state
 */
struct OpusCustomDecoder
{
    const OpusCustomMode *mode;
    int overlap;
    int channels;
    int stream_channels;

    int downsample;
    int start, end;
    int signalling;
    int arch;

    /* Everything beyond this point gets cleared on a reset */
#define DECODER_RESET_START rng

    opus_uint32 rng;
    int error;
    int last_pitch_index;
    int loss_count;
    int postfilter_period;
    int postfilter_period_old;
    opus_val16 postfilter_gain;
    opus_val16 postfilter_gain_old;
    int postfilter_tapset;
    int postfilter_tapset_old;

    celt_sig preemph_memD[2];

    celt_sig _decode_mem[1]; /* Size = channels*(DECODE_BUFFER_SIZE+mode->overlap) */
                             /* opus_val16 lpc[],  Size = channels*LPC_ORDER */
                             /* opus_val16 oldEBands[], Size = 2*mode->nbEBands */
                             /* opus_val16 oldLogE[], Size = 2*mode->nbEBands */
                             /* opus_val16 oldLogE2[], Size = 2*mode->nbEBands */
                             /* opus_val16 backgroundLogE[], Size = 2*mode->nbEBands */
};

OPUS_CUSTOM_NOSTATIC int opus_custom_decoder_get_size(const CELTMode *mode, int channels)
{
    int size = sizeof(struct CELTDecoder) + (channels * (DECODE_BUFFER_SIZE + mode->overlap) - 1) * sizeof(celt_sig) + channels * LPC_ORDER * sizeof(opus_val16) + 4 * 2 * mode->nbEBands * sizeof(opus_val16);
    return size;
}

int celt_decoder_get_size(int channels)
{
    const CELTMode *mode = opus_custom_mode_create(48000, 960, NULL);
    return opus_custom_decoder_get_size(mode, channels);
}

#ifdef CUSTOM_MODES
CELTDecoder *opus_custom_decoder_create(const CELTMode *mode, int channels, int *error)
{
    int ret;
    CELTDecoder *st = (CELTDecoder *)opus_alloc(opus_custom_decoder_get_size(mode, channels));
    ret = opus_custom_decoder_init(st, mode, channels);
    if (ret != OPUS_OK)
    {
        opus_custom_decoder_destroy(st);
        st = NULL;
    }
    if (error)
        *error = ret;
    return st;
}
#endif /* CUSTOM_MODES */

OPUS_CUSTOM_NOSTATIC int opus_custom_decoder_init(CELTDecoder *st, const CELTMode *mode, int channels)
{
    if (channels < 0 || channels > 2)
        return OPUS_BAD_ARG;

    if (st == NULL)
        return OPUS_ALLOC_FAIL;

    OPUS_CLEAR((char *)st, opus_custom_decoder_get_size(mode, channels));

    st->mode = mode;
    st->overlap = mode->overlap;
    st->stream_channels = st->channels = channels;

    st->downsample = 1;
    st->start = 0;
    st->end = st->mode->effEBands;
    st->signalling = 1;
    st->arch = opus_select_arch();

    st->loss_count = 0;

    opus_custom_decoder_ctl(st, OPUS_RESET_STATE);

    return OPUS_OK;
}

int celt_decoder_init(CELTDecoder *st, opus_int32 sampling_rate, int channels)
{
    int ret;
    ret = opus_custom_decoder_init(st, opus_custom_mode_create(48000, 960, NULL), channels);
    if (ret != OPUS_OK)
        return ret;
    st->downsample = resampling_factor(sampling_rate);
    if (st->downsample == 0)
        return OPUS_BAD_ARG;
    else
        return OPUS_OK;
}

#ifdef CUSTOM_MODES
void opus_custom_decoder_destroy(CELTDecoder *st)
{
    opus_free(st);
}
#endif /* CUSTOM_MODES */

static OPUS_INLINE opus_val16 SIG2WORD16(celt_sig x)
{
#ifdef FIXED_POINT
    x = PSHR32(x, SIG_SHIFT);
    x = MAX32(x, -32768);
    x = MIN32(x, 32767);
    return EXTRACT16(x);
#else
    return (opus_val16)x;
#endif
}

#ifndef RESYNTH
static
#endif
    void
    deemphasis(celt_sig *in[], opus_val16 *pcm, int N, int C, int downsample, const opus_val16 *coef, celt_sig *mem, celt_sig *OPUS_RESTRICT scratch)
{
    int c;
    int Nd;
    int apply_downsampling = 0;
    opus_val16 coef0;

    coef0 = coef[0];
    Nd = N / downsample;
    c = 0;
    do
    {
        int j;
        celt_sig *OPUS_RESTRICT x;
        opus_val16 *OPUS_RESTRICT y;
        celt_sig m = mem[c];
        x = in[c];
        y = pcm + c;
#ifdef CUSTOM_MODES
        if (coef[1] != 0)
        {
            opus_val16 coef1 = coef[1];
            opus_val16 coef3 = coef[3];
            for (j = 0; j < N; j++)
            {
                celt_sig tmp = x[j] + m + VERY_SMALL;
                m = MULT16_32_Q15(coef0, tmp) - MULT16_32_Q15(coef1, x[j]);
                tmp = SHL32(MULT16_32_Q15(coef3, tmp), 2);
                scratch[j] = tmp;
            }
            apply_downsampling = 1;
        }
        else
#endif
            if (downsample > 1)
        {
            /* Shortcut for the standard (non-custom modes) case */
            for (j = 0; j < N; j++)
            {
                celt_sig tmp = x[j] + m + VERY_SMALL;
                m = MULT16_32_Q15(coef0, tmp);
                scratch[j] = tmp;
            }
            apply_downsampling = 1;
        }
        else
        {
            /* Shortcut for the standard (non-custom modes) case */
            for (j = 0; j < N; j++)
            {
                celt_sig tmp = x[j] + m + VERY_SMALL;
                m = MULT16_32_Q15(coef0, tmp);
                y[j * C] = SCALEOUT(SIG2WORD16(tmp));
            }
        }
        mem[c] = m;

        if (apply_downsampling)
        {
            /* Perform down-sampling */
            for (j = 0; j < Nd; j++)
                y[j * C] = SCALEOUT(SIG2WORD16(scratch[j * downsample]));
        }
    } while (++c < C);
}

/** Compute the IMDCT and apply window for all sub-frames and
    all channels in a frame */
#ifndef RESYNTH
static
#endif
void
compute_inv_mdcts(const CELTMode *mode, int shortBlocks, celt_sig *X,
                    celt_sig *OPUS_RESTRICT out_mem[], int C, int LM)
{
    int b, c;
    int B;
    int N;
    int shift;
    const int overlap = OVERLAP(mode);

    if (shortBlocks)
    {
        B = shortBlocks;
        N = mode->shortMdctSize;
        shift = mode->maxLM;
    }
    else
    {
        B = 1;
        N = mode->shortMdctSize << LM;
        shift = mode->maxLM - LM;
    }

    if (B == 1 && C == 2)
    {
        float *in[2] = {(float *)X, (float *)(X + N)};
        float *out[2] = {(float *)out_mem[0], (float *)out_mem[1]};
        clt_mdct_backward_B1_C2(&mode->mdct, in, out, mode->window, overlap, shift, 1);
    }
    else if (B == 8 && C == 2)
    {
        for (b = 0; b < B; b++)
        {
            float *in[2] = {(float *)X + b + 0 * N * B, (float *)(X + b + 1 * N * B)};
            float *out[2] = {(float *)out_mem[0] + N * b, (float *)out_mem[1] + N * b};
            clt_mdct_backward_B1_C2(&mode->mdct, in, out, mode->window, overlap, shift, B);
        }
    }
    else
    {
        celt_assert(0);
        c = 0;
        do
        {
            /* IMDCT on the interleaved the sub-frames, overlap-add is performed by the IMDCT */
            for (b = 0; b < B; b++)
                clt_mdct_backward(&mode->mdct, &X[b + c * N * B], out_mem[c] + N * b, mode->window, overlap, shift, B);
        } while (++c < C);
    }
}

static void tf_decode(int start, int end, int isTransient, int *tf_res, int LM, ec_dec *dec)
{
    int i, curr, tf_select;
    int tf_select_rsv;
    int tf_changed;
    int logp;
    opus_uint32 budget;
    opus_uint32 tell;

    budget = dec->storage * 8;
    tell = ec_tell(dec);
    logp = isTransient ? 2 : 4;
    tf_select_rsv = LM > 0 && tell + logp + 1 <= budget;
    budget -= tf_select_rsv;
    tf_changed = curr = 0;
    for (i = start; i < end; i++)
    {
        if (tell + logp <= budget)
        {
            curr ^= ec_dec_bit_logp(dec, logp);
            tell = ec_tell(dec);
            tf_changed |= curr;
        }
        tf_res[i] = curr;
        logp = isTransient ? 4 : 5;
    }
    tf_select = 0;
    if (tf_select_rsv &&
        tf_select_table[LM][4 * isTransient + 0 + tf_changed] !=
            tf_select_table[LM][4 * isTransient + 2 + tf_changed])
    {
        tf_select = ec_dec_bit_logp(dec, 1);
    }
    for (i = start; i < end; i++)
    {
        tf_res[i] = tf_select_table[LM][4 * isTransient + 2 * tf_select + tf_res[i]];
    }
}

int celt_decode_with_ec(CELTDecoder *OPUS_RESTRICT st, const unsigned char *data, int len, opus_val16 *OPUS_RESTRICT pcm, int frame_size, ec_dec *dec)
{
    int c, i, N;
    int spread_decision;
    opus_int32 bits;
    ec_dec _dec;
    VARDECL(celt_sig, freq);
    VARDECL(celt_norm, X);
    VARDECL(int, fine_quant);
    VARDECL(int, pulses);
    VARDECL(int, cap);
    VARDECL(int, offsets);
    VARDECL(int, fine_priority);
    VARDECL(int, tf_res);
    VARDECL(unsigned char, collapse_masks);
    celt_sig *decode_mem[2];
    celt_sig *out_syn[2];
    opus_val16 *lpc;
    opus_val16 *oldBandE, *oldLogE, *oldLogE2, *backgroundLogE;

    int shortBlocks;
    int isTransient;
    int intra_ener;
    const int CC = st->channels;
    int LM, M;
    int effEnd;
    int codedBands;
    int alloc_trim;
    int postfilter_pitch;
    opus_val16 postfilter_gain;
    int intensity = 0;
    int dual_stereo = 0;
    opus_int32 total_bits;
    opus_int32 balance;
    opus_int32 tell;
    int dynalloc_logp;
    int postfilter_tapset;
    int anti_collapse_rsv;
    int anti_collapse_on = 0;
    int silence;
    int C = st->stream_channels;
    const OpusCustomMode *mode;
    int nbEBands;
    int overlap;
    const opus_int16 *eBands;
    ALLOC_STACK;

    mode = st->mode;
    nbEBands = mode->nbEBands;
    overlap = mode->overlap;
    eBands = mode->eBands;
    frame_size *= st->downsample;

    c = 0;
    do
    {
        decode_mem[c] = st->_decode_mem + c * (DECODE_BUFFER_SIZE + overlap);
    } while (++c < CC);
    lpc = (opus_val16 *)(st->_decode_mem + (DECODE_BUFFER_SIZE + overlap) * CC);
    oldBandE = lpc + CC * LPC_ORDER;
    oldLogE = oldBandE + 2 * nbEBands;
    oldLogE2 = oldLogE + 2 * nbEBands;
    backgroundLogE = oldLogE2 + 2 * nbEBands;

#ifdef CUSTOM_MODES
    if (st->signalling && data != NULL)
    {
        int data0 = data[0];
        /* Convert "standard mode" to Opus header */
        if (mode->Fs == 48000 && mode->shortMdctSize == 120)
        {
            data0 = fromOpus(data0);
            if (data0 < 0)
                return OPUS_INVALID_PACKET;
        }
        st->end = IMAX(1, mode->effEBands - 2 * (data0 >> 5));
        LM = (data0 >> 3) & 0x3;
        C = 1 + ((data0 >> 2) & 0x1);
        data++;
        len--;
        if (LM > mode->maxLM)
            return OPUS_INVALID_PACKET;
        if (frame_size < mode->shortMdctSize << LM)
            return OPUS_BUFFER_TOO_SMALL;
        else
            frame_size = mode->shortMdctSize << LM;
    }
    else
    {
#else
    {
#endif
        for (LM = 0; LM <= mode->maxLM; LM++)
            if (mode->shortMdctSize << LM == frame_size)
                break;
        if (LM > mode->maxLM)
            return OPUS_BAD_ARG;
    }
    M = 1 << LM;

    if (len < 0 || len > 1275 || pcm == NULL)
        return OPUS_BAD_ARG;

    N = M * mode->shortMdctSize;

    effEnd = st->end;
    if (effEnd > mode->effEBands)
        effEnd = mode->effEBands;

    if (dec == NULL)
    {
        ec_dec_init(&_dec, (unsigned char *)data, len);
        dec = &_dec;
    }

    if (C == 1)
    {
        for (i = 0; i < nbEBands; i++)
            oldBandE[i] = MAX16(oldBandE[i], oldBandE[nbEBands + i]);
    }

    total_bits = len * 8;
    tell = ec_tell(dec);

    if (tell >= total_bits)
        silence = 1;
    else if (tell == 1)
        silence = ec_dec_bit_logp(dec, 15);
    else
        silence = 0;
    if (silence)
    {
        /* Pretend we've read all the remaining bits */
        tell = len * 8;
        dec->nbits_total += tell - ec_tell(dec);
    }

    postfilter_gain = 0;
    postfilter_pitch = 0;
    postfilter_tapset = 0;
    if (st->start == 0 && tell + 16 <= total_bits)
    {
        if (ec_dec_bit_logp(dec, 1))
        {
            int qg, octave;
            octave = ec_dec_uint(dec, 6);
            postfilter_pitch = (16 << octave) + ec_dec_bits(dec, 4 + octave) - 1;
            qg = ec_dec_bits(dec, 3);
            if (ec_tell(dec) + 2 <= total_bits)
                postfilter_tapset = ec_dec_icdf(dec, tapset_icdf, 2);
            postfilter_gain = QCONST16(.09375f, 15) * (qg + 1);
        }
        tell = ec_tell(dec);
    }

    if (LM > 0 && tell + 3 <= total_bits)
    {
        isTransient = ec_dec_bit_logp(dec, 3);
        tell = ec_tell(dec);
    }
    else
        isTransient = 0;

    if (isTransient)
        shortBlocks = M;
    else
        shortBlocks = 0;

    /* Decode the global flags (first symbols in the stream) */
    intra_ener = tell + 3 <= total_bits ? ec_dec_bit_logp(dec, 3) : 0;
    /* Get band energies */
    unquant_coarse_energy(mode, st->start, st->end, oldBandE,
                          intra_ener, dec, C, LM);

    ALLOC(tf_res, nbEBands, int);
    tf_decode(st->start, st->end, isTransient, tf_res, LM, dec);

    tell = ec_tell(dec);
    spread_decision = SPREAD_NORMAL;
    if (tell + 4 <= total_bits)
        spread_decision = ec_dec_icdf(dec, spread_icdf, 5);

    ALLOC(cap, nbEBands, int);

    init_caps(mode, cap, LM, C);

    ALLOC(offsets, nbEBands, int);

    dynalloc_logp = 6;
    total_bits <<= BITRES;
    tell = ec_tell_frac(dec);
    for (i = st->start; i < st->end; i++)
    {
        int width, quanta;
        int dynalloc_loop_logp;
        int boost;
        width = C * (eBands[i + 1] - eBands[i]) << LM;
        /* quanta is 6 bits, but no more than 1 bit/sample
           and no less than 1/8 bit/sample */
        quanta = IMIN(width << BITRES, IMAX(6 << BITRES, width));
        dynalloc_loop_logp = dynalloc_logp;
        boost = 0;
        while (tell + (dynalloc_loop_logp << BITRES) < total_bits && boost < cap[i])
        {
            int flag;
            flag = ec_dec_bit_logp(dec, dynalloc_loop_logp);
            tell = ec_tell_frac(dec);
            if (!flag)
                break;
            boost += quanta;
            total_bits -= quanta;
            dynalloc_loop_logp = 1;
        }
        offsets[i] = boost;
        /* Making dynalloc more likely */
        if (boost > 0)
            dynalloc_logp = IMAX(2, dynalloc_logp - 1);
    }

    ALLOC(fine_quant, nbEBands, int);
    alloc_trim = tell + (6 << BITRES) <= total_bits ? ec_dec_icdf(dec, trim_icdf, 7) : 5;

    bits = (((opus_int32)len * 8) << BITRES) - ec_tell_frac(dec) - 1;
    anti_collapse_rsv = isTransient && LM >= 2 && bits >= ((LM + 2) << BITRES) ? (1 << BITRES) : 0;
    bits -= anti_collapse_rsv;

    ALLOC(pulses, nbEBands, int);
    ALLOC(fine_priority, nbEBands, int);

    codedBands = compute_allocation(mode, st->start, st->end, offsets, cap,
                                    alloc_trim, &intensity, &dual_stereo, bits, &balance, pulses,
                                    fine_quant, fine_priority, C, LM, dec, 0, 0, 0);

    unquant_fine_energy(mode, st->start, st->end, oldBandE, fine_quant, dec, C);

    /* Decode fixed codebook */
    ALLOC(collapse_masks, C * nbEBands, unsigned char);
    ALLOC(X, C * N, celt_norm); /**< Interleaved normalised MDCTs */

    quant_all_bands(0, mode, st->start, st->end, X, C == 2 ? X + N : NULL, collapse_masks,
                    NULL, pulses, shortBlocks, spread_decision, dual_stereo, intensity, tf_res,
                    len * (8 << BITRES) - anti_collapse_rsv, balance, dec, LM, codedBands, &st->rng);

    if (anti_collapse_rsv > 0)
    {
        anti_collapse_on = ec_dec_bits(dec, 1);
    }

    unquant_energy_finalise(mode, st->start, st->end, oldBandE,
                            fine_quant, fine_priority, len * 8 - ec_tell(dec), dec, C);

    if (anti_collapse_on)
        anti_collapse(mode, X, collapse_masks, LM, C, N,
                      st->start, st->end, oldBandE, oldLogE, oldLogE2, pulses, st->rng);

    ALLOC(freq, IMAX(CC, C) * N, celt_sig); /**< Interleaved signal MDCTs */

    if (silence)
    {
        for (i = 0; i < C * nbEBands; i++)
            oldBandE[i] = -QCONST16(28.f, DB_SHIFT);
        for (i = 0; i < C * N; i++)
            freq[i] = 0;
    }
    else
    {
        /* Synthesis */
        denormalise_bands(mode, X, freq, oldBandE, st->start, effEnd, C, M);
    }
    c = 0;
    do
    {
        OPUS_MOVE(decode_mem[c], decode_mem[c] + N, DECODE_BUFFER_SIZE - N + overlap / 2);
    } while (++c < CC);

    c = 0;
    do
    {
        int bound = M * eBands[effEnd];
        if (st->downsample != 1)
            bound = IMIN(bound, N / st->downsample);
        for (i = bound; i < N; i++)
            freq[c * N + i] = 0;
    } while (++c < C);

    c = 0;
    do
    {
        out_syn[c] = decode_mem[c] + DECODE_BUFFER_SIZE - N;
    } while (++c < CC);

    if (CC == 2 && C == 1)
    {
        for (i = 0; i < N; i++)
            freq[N + i] = freq[i];
    }
    if (CC == 1 && C == 2)
    {
        for (i = 0; i < N; i++)
            freq[i] = HALF32(ADD32(freq[i], freq[N + i]));
    }

    /* Compute inverse MDCTs */
    compute_inv_mdcts(mode, shortBlocks, freq, out_syn, CC, LM);

    c = 0;
    do
    {
        st->postfilter_period = IMAX(st->postfilter_period, COMBFILTER_MINPERIOD);
        st->postfilter_period_old = IMAX(st->postfilter_period_old, COMBFILTER_MINPERIOD);
        comb_filter(out_syn[c], out_syn[c], st->postfilter_period_old, st->postfilter_period, mode->shortMdctSize,
                    st->postfilter_gain_old, st->postfilter_gain, st->postfilter_tapset_old, st->postfilter_tapset,
                    mode->window, overlap);
        if (LM != 0)
            comb_filter(out_syn[c] + mode->shortMdctSize, out_syn[c] + mode->shortMdctSize, st->postfilter_period, postfilter_pitch, N - mode->shortMdctSize,
                        st->postfilter_gain, postfilter_gain, st->postfilter_tapset, postfilter_tapset,
                        mode->window, overlap);

    } while (++c < CC);
    st->postfilter_period_old = st->postfilter_period;
    st->postfilter_gain_old = st->postfilter_gain;
    st->postfilter_tapset_old = st->postfilter_tapset;
    st->postfilter_period = postfilter_pitch;
    st->postfilter_gain = postfilter_gain;
    st->postfilter_tapset = postfilter_tapset;
    if (LM != 0)
    {
        st->postfilter_period_old = st->postfilter_period;
        st->postfilter_gain_old = st->postfilter_gain;
        st->postfilter_tapset_old = st->postfilter_tapset;
    }

    if (C == 1)
    {
        for (i = 0; i < nbEBands; i++)
            oldBandE[nbEBands + i] = oldBandE[i];
    }

    /* In case start or end were to change */
    if (!isTransient)
    {
        for (i = 0; i < 2 * nbEBands; i++)
            oldLogE2[i] = oldLogE[i];
        for (i = 0; i < 2 * nbEBands; i++)
            oldLogE[i] = oldBandE[i];
        for (i = 0; i < 2 * nbEBands; i++)
            backgroundLogE[i] = MIN16(backgroundLogE[i] + M * QCONST16(0.001f, DB_SHIFT), oldBandE[i]);
    }
    else
    {
        for (i = 0; i < 2 * nbEBands; i++)
            oldLogE[i] = MIN16(oldLogE[i], oldBandE[i]);
    }
    c = 0;
    do
    {
        for (i = 0; i < st->start; i++)
        {
            oldBandE[c * nbEBands + i] = 0;
            oldLogE[c * nbEBands + i] = oldLogE2[c * nbEBands + i] = -QCONST16(28.f, DB_SHIFT);
        }
        for (i = st->end; i < nbEBands; i++)
        {
            oldBandE[c * nbEBands + i] = 0;
            oldLogE[c * nbEBands + i] = oldLogE2[c * nbEBands + i] = -QCONST16(28.f, DB_SHIFT);
        }
    } while (++c < 2);
    st->rng = dec->rng;

    /* We reuse freq[] as scratch space for the de-emphasis */
    deemphasis(out_syn, pcm, N, CC, st->downsample, mode->preemph, st->preemph_memD, freq);
    st->loss_count = 0;
    RESTORE_STACK;
    if (ec_tell(dec) > 8 * len)
        return OPUS_INTERNAL_ERROR;
    if (ec_get_error(dec))
        st->error = 1;
    return frame_size / st->downsample;
}

#ifdef CUSTOM_MODES

#ifdef FIXED_POINT
int opus_custom_decode(CELTDecoder *OPUS_RESTRICT st, const unsigned char *data, int len, opus_int16 *OPUS_RESTRICT pcm, int frame_size)
{
    return celt_decode_with_ec(st, data, len, pcm, frame_size, NULL);
}

#ifndef DISABLE_FLOAT_API
int opus_custom_decode_float(CELTDecoder *OPUS_RESTRICT st, const unsigned char *data, int len, float *OPUS_RESTRICT pcm, int frame_size)
{
    int j, ret, C, N;
    VARDECL(opus_int16, out);
    ALLOC_STACK;

    if (pcm == NULL)
        return OPUS_BAD_ARG;

    C = st->channels;
    N = frame_size;

    ALLOC(out, C * N, opus_int16);
    ret = celt_decode_with_ec(st, data, len, out, frame_size, NULL);
    if (ret > 0)
        for (j = 0; j < C * ret; j++)
            pcm[j] = out[j] * (1.f / 32768.f);

    RESTORE_STACK;
    return ret;
}
#endif /* DISABLE_FLOAT_API */

#else

int opus_custom_decode_float(CELTDecoder *OPUS_RESTRICT st, const unsigned char *data, int len, float *OPUS_RESTRICT pcm, int frame_size)
{
    return celt_decode_with_ec(st, data, len, pcm, frame_size, NULL);
}

int opus_custom_decode(CELTDecoder *OPUS_RESTRICT st, const unsigned char *data, int len, opus_int16 *OPUS_RESTRICT pcm, int frame_size)
{
    int j, ret, C, N;
    VARDECL(celt_sig, out);
    ALLOC_STACK;

    if (pcm == NULL)
        return OPUS_BAD_ARG;

    C = st->channels;
    N = frame_size;
    ALLOC(out, C * N, celt_sig);

    ret = celt_decode_with_ec(st, data, len, out, frame_size, NULL);

    if (ret > 0)
        for (j = 0; j < C * ret; j++)
            pcm[j] = FLOAT2INT16(out[j]);

    RESTORE_STACK;
    return ret;
}

#endif
#endif /* CUSTOM_MODES */

int opus_custom_decoder_ctl(CELTDecoder *OPUS_RESTRICT st, int request, ...)
{
    va_list ap;

    va_start(ap, request);
    switch (request)
    {
    case CELT_SET_START_BAND_REQUEST:
    {
        opus_int32 value = va_arg(ap, opus_int32);
        if (value < 0 || value >= st->mode->nbEBands)
            goto bad_arg;
        st->start = value;
    }
    break;
    case CELT_SET_END_BAND_REQUEST:
    {
        opus_int32 value = va_arg(ap, opus_int32);
        if (value < 1 || value > st->mode->nbEBands)
            goto bad_arg;
        st->end = value;
    }
    break;
    case CELT_SET_CHANNELS_REQUEST:
    {
        opus_int32 value = va_arg(ap, opus_int32);
        if (value < 1 || value > 2)
            goto bad_arg;
        st->stream_channels = value;
    }
    break;
    case CELT_GET_AND_CLEAR_ERROR_REQUEST:
    {
        opus_int32 *value = va_arg(ap, opus_int32 *);
        if (value == NULL)
            goto bad_arg;
        *value = st->error;
        st->error = 0;
    }
    break;
    case OPUS_GET_LOOKAHEAD_REQUEST:
    {
        opus_int32 *value = va_arg(ap, opus_int32 *);
        if (value == NULL)
            goto bad_arg;
        *value = st->overlap / st->downsample;
    }
    break;
    case OPUS_RESET_STATE:
    {
        int i;
        opus_val16 *lpc, *oldBandE, *oldLogE, *oldLogE2;
        lpc = (opus_val16 *)(st->_decode_mem + (DECODE_BUFFER_SIZE + st->overlap) * st->channels);
        oldBandE = lpc + st->channels * LPC_ORDER;
        oldLogE = oldBandE + 2 * st->mode->nbEBands;
        oldLogE2 = oldLogE + 2 * st->mode->nbEBands;
        OPUS_CLEAR((char *)&st->DECODER_RESET_START,
                   opus_custom_decoder_get_size(st->mode, st->channels) -
                       ((char *)&st->DECODER_RESET_START - (char *)st));
        for (i = 0; i < 2 * st->mode->nbEBands; i++)
            oldLogE[i] = oldLogE2[i] = -QCONST16(28.f, DB_SHIFT);
    }
    break;
    case OPUS_GET_PITCH_REQUEST:
    {
        opus_int32 *value = va_arg(ap, opus_int32 *);
        if (value == NULL)
            goto bad_arg;
        *value = st->postfilter_period;
    }
    break;
    case CELT_GET_MODE_REQUEST:
    {
        const CELTMode **value = va_arg(ap, const CELTMode **);
        if (value == 0)
            goto bad_arg;
        *value = st->mode;
    }
    break;
    case CELT_SET_SIGNALLING_REQUEST:
    {
        opus_int32 value = va_arg(ap, opus_int32);
        st->signalling = value;
    }
    break;
    case OPUS_GET_FINAL_RANGE_REQUEST:
    {
        opus_uint32 *value = va_arg(ap, opus_uint32 *);
        if (value == 0)
            goto bad_arg;
        *value = st->rng;
    }
    break;
    default:
        goto bad_request;
    }
    va_end(ap);
    return OPUS_OK;
bad_arg:
    va_end(ap);
    return OPUS_BAD_ARG;
bad_request:
    va_end(ap);
    return OPUS_UNIMPLEMENTED;
}
