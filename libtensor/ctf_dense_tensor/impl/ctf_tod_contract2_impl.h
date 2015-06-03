#ifndef LIBTENSOR_CTF_TOD_CONTRACT2_IMPL_H
#define LIBTENSOR_CTF_TOD_CONTRACT2_IMPL_H

#include <libtensor/core/bad_dimensions.h>
#include <libtensor/core/mask.h>
#include "../ctf_dense_tensor_ctrl.h"
#include "../ctf_error.h"
#include "../ctf_dense_tensor.h"
#include "../ctf_tod_contract2.h"
#include "ctf_tod_aux_symcomp.h"

namespace libtensor {


template<size_t N, size_t M, size_t K>
const char ctf_tod_contract2<N, M, K>::k_clazz[] =
    "ctf_tod_contract2<N, M, K>";


template<size_t N, size_t M, size_t K>
ctf_tod_contract2<N, M, K>::ctf_tod_contract2(
    const contraction2<N, M, K> &contr,
    ctf_dense_tensor_i<NA, double> &ta,
    ctf_dense_tensor_i<NB, double> &tb,
    double d) :

    m_contr(contr), m_ta(ta), m_tb(tb),
    m_dimsc(contr, ta.get_dims(), tb.get_dims()), m_d(d) {

}


namespace {

template<size_t N, size_t M, size_t K>
void ctf_tod_contract2_make_map(
    const contraction2<N, M, K> &contr,
    char (&map)[2 * (N + M + K)]) {

    enum {
        NA = N + K, NB = M + K, NC = N + M
    };

    const sequence<NA + NB + NC, size_t> &conn = contr.get_conn();
    for(size_t i = 0; i < NC; i++) {
        map[i] = i + 1;
        size_t ii = conn[NC - i - 1] - NC;
        if(ii < NA) {
            ii = NA - ii - 1;
            map[NC + ii] = i + 1;
        } else {
            ii = NB + NA - ii - 1;
            map[NC + NA + ii] = i + 1;
        }
    }
    for(size_t i = 0, j = NC; i < NA; i++) {
        size_t ii = NC + i;
        if(conn[ii] >= NC + NA) {
            size_t ia = NA - i - 1;
            size_t ib = NC + NA + NB - conn[ii] - 1;
            map[NC + ia] = j + 1;
            map[NC + NA + ib] = j + 1;
            j++;
        }
    }
}

template<size_t N, size_t M, size_t K>
bool ctf_tod_contract2_symmetry(const contraction2<N, M, K> &contr,
    const ctf_symmetry<N + K, double> &syma, size_t icompa,
    const ctf_symmetry<M + K, double> &symb, size_t icompb,
    ctf_symmetry<N + M, double> &symc) {

    enum {
        NA = N + K, NB = M + K, NC = N + M
    };

    const sequence<NA + NB + NC, size_t> &conn = contr.get_conn();
    const sequence<NA, unsigned> &grpa = syma.get_grp(icompa);
    const sequence<NA, unsigned> &taga = syma.get_sym(icompa);
    const sequence<NB, unsigned> &grpb = symb.get_grp(icompb);
    const sequence<NB, unsigned> &tagb = symb.get_sym(icompb);

    mask<NA> mappeda;
    mask<NB> mappedb;
    sequence<NA, size_t> mapa;
    sequence<NB, size_t> mapb;

    //  Determine whether the contraction goes over a symmetric and
    //  antisymmetric group, in which case it yields zero and can be
    //  skipped

    bool zero_skip = false;
    sequence<K, size_t> za, zb;
    mask<K> mskz;
    for(size_t i = 0, j = 0; i < NA; i++) if(conn[NC + i] >= NC + NA) {
        za[j] = i;
        zb[j] = conn[NC + i] - NC - NA;
        j++;
    }
    for(size_t i = 0; i < K; i++) if(!mskz[i]) {
        unsigned ga = grpa[za[i]], gb = grpb[zb[i]];
        mask<K> ma, mb;
        for(size_t j = 0; j < K; j++) {
            ma[j] = (grpa[za[j]] == ga);
            mb[j] = (grpb[zb[j]] == gb);
        }
        mask<K> mab = ma & mb;
        size_t nm = 0;
        for(size_t j = 0; j < K; j++) if(mab[j]) nm++;
        if(nm > 1 && taga[ga] != tagb[gb]) zero_skip = true;
    }
    if(zero_skip) return false;

    //  If the contraction is nonzero, figure output symmetry

    sequence<NC, unsigned> grpc, tagc;
    unsigned igrp = 0;
    for(size_t i = 0; i < NA; i++) if(conn[NC + i] < NC) {
        if(!mappeda[i]) {
            for(size_t j = i; j < NA; j++) if(grpa[i] == grpa[j]) {
                mapa[grpa[j]] = igrp;
                mappeda[j] = true;
            }
            tagc[igrp] = taga[grpa[i]];
            igrp++;
        }
        grpc[conn[NC + i]] = mapa[grpa[i]];
    }
    for(size_t i = 0; i < NB; i++) if(conn[NC + NA + i] < NC) {
        if(!mappedb[i]) {
            for(size_t j = i; j < NB; j++) if(grpb[i] == grpb[j]) {
                mapb[grpb[j]] = igrp;
                mappedb[j] = true;
            }
            tagc[igrp] = tagb[grpb[i]];
            igrp++;
        }
        grpc[conn[NC + NA + i]] = mapb[grpb[i]];
    }

    symc = ctf_symmetry<NC, double>(grpc, tagc);
    return true;
}

} // unnamed namespace


template<size_t N, size_t M, size_t K>
void ctf_tod_contract2<N, M, K>::perform(
    bool zero,
    ctf_dense_tensor_i<NC, double> &tc) {

    static const char method[] =
        "perform(bool, ctf_dense_tensor_i<NC, double>&)";

    if(!m_dimsc.get_dims().equals(tc.get_dims())) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "tc");
    }

    ctf_dense_tensor_ctrl<NA, double> ca(m_ta);
    ctf_dense_tensor_ctrl<NB, double> cb(m_tb);
    ctf_dense_tensor_ctrl<NC, double> cc(tc);

    const ctf_symmetry<NA, double> &syma = ca.req_symmetry();
    const ctf_symmetry<NB, double> &symb = cb.req_symmetry();
    const ctf_symmetry<NC, double> &symc = cc.req_symmetry();

    char map[NC + NA + NB];
    ctf_tod_contract2_make_map(m_contr, map);

    ctf_tod_contract2::start_timer();

    if(zero) {
        for(size_t i = 0; i < symc.get_ncomp(); i++) {
            CTF::Tensor<double> &dtc = cc.req_ctf_tensor(i);
            dtc = 0.0;
        }
    }
    for(size_t icompa = 0; icompa < syma.get_ncomp(); icompa++)
    for(size_t icompb = 0; icompb < symb.get_ncomp(); icompb++) {

        ctf_symmetry<NC, double> symab;
        bool nonzero = ctf_tod_contract2_symmetry(m_contr, syma, icompa,
            symb, icompb, symab);
        if(!nonzero) continue;

        size_t icompc = ctf_tod_aux_symcomp(symab, 0, symc);
        double z = ctf_symmetry<NC, double>::symconv_factor(symab, 0,
            symc, icompc);

        int sab[NC], sc[NC];
        symab.write(0, sab);
        symc.write(icompc, sc);
        bool use_interm = false;
        for(size_t i = 0; i < NC; i++) if(sab[i] != sc[i]) use_interm = true;

        CTF::Tensor<double> &dta = ca.req_ctf_tensor(icompa);
        CTF::Tensor<double> &dtb = cb.req_ctf_tensor(icompb);
        CTF::Tensor<double> &dtc = cc.req_ctf_tensor(icompc);

        if(use_interm) {
            ctf_dense_tensor<NC, double> tx(tc.get_dims(), symab);
            ctf_dense_tensor_ctrl<NC, double> cx(tx);
            CTF::Tensor<double> &dtx = cx.req_ctf_tensor(0);
            dtx.contract(1.0, dta, &map[NC], dtb, &map[NC + NA],
                0.0, &map[0]);
            dtc.sum(m_d * z, dtx, &map[NC], 1.0, &map[NC]);
        } else {
            dtc.contract(m_d * z, dta, &map[NC], dtb, &map[NC + NA], 1.0,
                &map[0]);
        }
    }

    ctf_tod_contract2::stop_timer();
}


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_CONTRACT2_IMPL_H

