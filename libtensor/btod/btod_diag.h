#ifndef LIBTENSOR_BTOD_DIAG_H
#define LIBTENSOR_BTOD_DIAG_H

#include "../defs.h"
#include "../not_implemented.h"
#include "../core/abs_index.h"
#include "../core/block_index_subspace_builder.h"
#include "../core/block_tensor_i.h"
#include "../core/block_tensor_ctrl.h"
#include "../core/orbit.h"
#include "../core/orbit_list.h"
#include "../core/permutation_builder.h"
#include "../symmetry/so_merge.h"
#include "../symmetry/so_permute.h"
#include "../tod/tod_copy.h"
#include "../tod/tod_diag.h"
#include "../tod/tod_set.h"
#include "bad_block_index_space.h"
#include "additive_btod.h"

namespace libtensor {


/** \brief Extracts a general diagonal from a block %tensor
    \tparam N Tensor order.
    \tparam M Diagonal order.

    \ingroup libtensor_btod
 **/
template<size_t N, size_t M>
class btod_diag :
    public additive_btod<N - M + 1>,
    public timings< btod_diag<N, M> > {

public:
    static const char *k_clazz; //!< Class name

public:
    static const size_t k_ordera = N; //!< Order of the argument
    static const size_t k_orderb = N - M + 1; //!< Order of the result

private:
    block_tensor_i<N, double> &m_bta; //!< Input block %tensor
    mask<N> m_msk; //!< Diagonal %mask
    permutation<k_orderb> m_perm; //!< Permutation of the result
    double m_c; //!< Scaling coefficient
    block_index_space<k_orderb> m_bis; //!< Block %index space of the result
    symmetry<k_orderb, double> m_sym; //!< Symmetry of the result
    assignment_schedule<k_orderb, double> m_sch; //!< Assignment schedule

public:
    //!    \name Construction and destruction
    //@{

    /** \brief Initializes the diagonal extraction operation
        \param bta Input block %tensor
        \param msk Mask which specifies the indexes to take the diagonal
        \param c Scaling factor
     **/
    btod_diag(block_tensor_i<N, double> &bta, const mask<N> &m,
        double c = 1.0);

    /** \brief Initializes the diagonal extraction operation
        \param bta Input block %tensor
        \param msk Mask which specifies the indexes to take the diagonal
        \param p Permutation of result tensor
        \param c Scaling factor
     **/
    btod_diag(block_tensor_i<N, double> &bta, const mask<N> &m,
        const permutation<N - M + 1> &p, double c = 1.0);

    //@}

    //!    \name Implementation of
    //      libtensor::direct_tensor_operation<N - M + 1, double>
    //@{

    virtual const block_index_space<k_orderb> &get_bis() const {
        return m_bis;
    }

    virtual const symmetry<k_orderb, double> &get_symmetry() const {
        return m_sym;
    }

    virtual const assignment_schedule<k_orderb, double> &get_schedule() const {
        return m_sch;
    }

    virtual void sync_on();
    virtual void sync_off();

    //@}

    using additive_btod<k_orderb>::perform;

protected:
    virtual void compute_block(bool zero, dense_tensor_i<k_orderb, double> &blk,
        const index<k_orderb> &ib, const tensor_transf<k_orderb, double> &trb,
        double c, cpu_pool &cpus);

private:
    /** \brief Forms the block %index space of the output or throws an
            exception if the input is incorrect.
     **/
    static block_index_space<N - M + 1> mk_bis(
        const block_index_space<N> &bis, const mask<N> &msk);

    /** \brief Sets up the symmetry of the operation result
     **/
    void make_symmetry();

    /** \brief Sets up the assignment schedule for the operation.
     **/
    void make_schedule();

    void compute_block(dense_tensor_i<k_orderb, double> &blk,
        const index<k_orderb> &ib, const tensor_transf<k_orderb, double> &trb,
        bool zero, double c, cpu_pool &cpus);

private:
    btod_diag(const btod_diag<N, M>&);
    const btod_diag<N, M> &operator=(const btod_diag<N, M>&);

};


template<size_t N, size_t M>
const char *btod_diag<N, M>::k_clazz = "btod_diag<N, M>";


template<size_t N, size_t M>
btod_diag<N, M>::btod_diag(block_tensor_i<N, double> &bta, const mask<N> &m,
    double c) :

    m_bta(bta), m_msk(m), m_c(c),
    m_bis(mk_bis(bta.get_bis(), m_msk)),
    m_sym(m_bis), m_sch(m_bis.get_block_index_dims()) {

    make_symmetry();
    make_schedule();
}


template<size_t N, size_t M>
btod_diag<N, M>::btod_diag(block_tensor_i<N, double> &bta, const mask<N> &m,
    const permutation<N - M + 1> &p, double c) :

    m_bta(bta), m_msk(m), m_perm(p), m_c(c),
    m_bis(mk_bis(bta.get_bis(), m_msk).permute(p)),
    m_sym(m_bis), m_sch(m_bis.get_block_index_dims())  {

    make_symmetry();
    make_schedule();
}


template<size_t N, size_t M>
void btod_diag<N, M>::sync_on() {

    block_tensor_ctrl<N, double> ctrla(m_bta);
    ctrla.req_sync_on();
}


template<size_t N, size_t M>
void btod_diag<N, M>::sync_off() {

    block_tensor_ctrl<N, double> ctrla(m_bta);
    ctrla.req_sync_off();
}

/*
template<size_t N, size_t M>
void btod_diag<N, M>::compute_block(dense_tensor_i<k_orderb, double> &blk,
    const index<k_orderb> &ib) {

    tensor_transf<k_orderb, double> trb0;
    compute_block(blk, ib, trb0, true, 1.0);
}*/


template<size_t N, size_t M>
void btod_diag<N, M>::compute_block(bool zero,
        dense_tensor_i<k_orderb, double> &blk, const index<k_orderb> &ib,
        const tensor_transf<k_orderb, double> &trb, double c, cpu_pool &cpus) {

    compute_block(blk, ib, trb, zero, c, cpus);
}


template<size_t N, size_t M>
void btod_diag<N, M>::compute_block(dense_tensor_i<k_orderb, double> &blk,
    const index<k_orderb> &ib, const tensor_transf<k_orderb, double> &trb,
    bool zero, double c, cpu_pool &cpus) {

    btod_diag<N, M>::start_timer();

    try {

        block_tensor_ctrl<N, double> ctrla(m_bta);
        dimensions<N> bidimsa = m_bta.get_bis().get_block_index_dims();

        //  Build ia from ib
        //
        sequence<k_ordera, size_t> map(0);
        size_t j = 0, jd; // Current index, index on diagonal
        bool b = false;
        for(size_t i = 0; i < k_ordera; i++) {
            if(m_msk[i]) {
                if(!b) { map[i] = jd = j++; b = true; }
                else { map[i] = jd; }
            } else {
                map[i] = j++;
            }
        }
        index<k_ordera> ia;
        index<k_orderb> ib2(ib);
        permutation<k_orderb> pinvb(m_perm, true);
        ib2.permute(pinvb);
        for(size_t i = 0; i < k_ordera; i++) ia[i] = ib2[map[i]];

        //  Find canonical index cia, transformation cia->ia
        //
        orbit<k_ordera, double> oa(ctrla.req_const_symmetry(), ia);
        abs_index<k_ordera> acia(oa.get_abs_canonical_index(), bidimsa);
        const tensor_transf<k_ordera, double> &tra = oa.get_transf(ia);
        permutation<k_ordera> pinva(tra.get_perm(), true);

        //  Build new diagonal mask and permutation in b
        //
        mask<k_ordera> m1(m_msk), m2(m_msk);
        sequence<k_ordera, size_t> map1(map), map2(map);
        m2.permute(pinva);
        pinva.apply(map2);

        sequence<N - M, size_t> seq1(0), seq2(0);
        sequence<k_orderb, size_t> seqb1(0), seqb2(0);
        for(register size_t i = 0, j1 = 0, j2 = 0; i < k_ordera; i++) {
            if(!m1[i]) seq1[j1++] = map1[i];
            if(!m2[i]) seq2[j2++] = map2[i];
        }
        bool b1 = false, b2 = false;
        for(register size_t i = 0, j1 = 0, j2 = 0; i < k_orderb; i++) {
            if(m1[i] && !b1) { seqb1[i] = k_orderb; b1 = true; }
            else { seqb1[i] = seq1[j1++]; }
            if(m2[i] && !b2) { seqb2[i] = k_orderb; b2 = true; }
            else { seqb2[i] = seq2[j2++]; }
        }

        permutation_builder<k_orderb> pb(seqb2, seqb1);
        permutation<k_orderb> permb(pb.get_perm());
        permb.permute(m_perm);
        permb.permute(trb.get_perm());

        //  Invoke the tensor operation
        //
        dense_tensor_i<k_ordera, double> &blka = ctrla.req_block(acia.get_index());
        double k = m_c * c * trb.get_scalar_tr().get_coeff()
                / tra.get_scalar_tr().get_coeff();
        if(zero) tod_diag<N, M>(blka, m2, permb, k).perform(blk);
        else tod_diag<N, M>(blka, m2, permb, k).perform(blk, 1.0);
        ctrla.ret_block(acia.get_index());

    }
    catch (...) {
        btod_diag<N, M>::stop_timer();
        throw;
    }

    btod_diag<N, M>::stop_timer();

}


template<size_t N, size_t M>
block_index_space<N - M + 1> btod_diag<N, M>::mk_bis(
    const block_index_space<N> &bis, const mask<N> &msk) {

    static const char *method =
        "mk_bis(const block_index_space<N>&, const mask<N>&)";

    //  Create the mask for the subspace builder
    //
    mask<N> m;
    bool b = false;
    for(size_t i = 0; i < N; i++) {
        if(msk[i]) {
            if(!b) { m[i] = true; b = true; }
        } else {
            m[i] = true;
        }
    }

    //  Build the output block index space
    //
    block_index_subspace_builder<N - M + 1, M - 1> bb(bis, m);
    block_index_space<k_orderb> obis(bb.get_bis());
    obis.match_splits();

    return obis;
}


template<size_t N, size_t M>
void btod_diag<N, M>::make_symmetry() {

    block_tensor_ctrl<k_ordera, double> ca(m_bta);

    block_index_space<k_orderb> bis(m_bis);
    permutation<k_orderb> pinv(m_perm, true);
    bis.permute(pinv);
    symmetry<k_orderb, double> symx(bis);
    so_merge<N, M - 1, double>(ca.req_const_symmetry(),
            m_msk, sequence<N, size_t>()).perform(symx);
    so_permute<k_orderb, double>(symx, m_perm).perform(m_sym);

}


template<size_t N, size_t M>
void btod_diag<N, M>::make_schedule() {

    block_tensor_ctrl<N, double> ctrla(m_bta);
    dimensions<N> bidimsa = m_bta.get_bis().get_block_index_dims();

    permutation<k_orderb> pinv(m_perm, true);
    size_t map[k_ordera];
    size_t j = 0, jd;
    bool b = false;
    for(size_t i = 0; i < k_ordera; i++) {
        if(m_msk[i]) {
            if(b) map[i] = jd;
            else { map[i] = jd = j++; b = true; }
        } else {
            map[i] = j++;
        }
    }

    orbit_list<k_ordera, double> ola(ctrla.req_const_symmetry());
    orbit_list<k_orderb, double> olb(m_sym);
    for (typename orbit_list<k_orderb, double>::iterator iob = olb.begin();
            iob != olb.end(); iob++) {

        index<k_ordera> idxa;
        index<k_orderb> idxb(olb.get_index(iob));
        idxb.permute(pinv);

        for(size_t i = 0; i < k_ordera; i++) idxa[i] = idxb[map[i]];

        orbit<k_ordera, double> oa(ctrla.req_const_symmetry(), idxa);
        if(! ola.contains(oa.get_abs_canonical_index())) continue;

        abs_index<k_ordera> cidxa(oa.get_abs_canonical_index(), bidimsa);

        if(ctrla.req_is_zero_block(cidxa.get_index())) continue;

        m_sch.insert(olb.get_abs_index(iob));
    }
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_DIAG_H
