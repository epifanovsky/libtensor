#ifndef LIBTENSOR_BTOD_CONTRACT2_H
#define LIBTENSOR_BTOD_CONTRACT2_H

#include <list>
#include <map>
#include <vector>
#include <libtensor/timings.h>
#include <libtensor/tod/contraction2.h>
#include <libtensor/block_tensor/block_tensor_i.h>
#include <libtensor/block_tensor/bto/additive_bto.h>
#include <libtensor/block_tensor/btod/btod_traits.h>
#include <libtensor/block_tensor/bto/bto_contract2_sym.h>
#include <libtensor/block_tensor/bto/bto_stream_i.h>

namespace libtensor {


template<size_t N, size_t M, size_t K>
struct btod_contract2_clazz {
    static const char *k_clazz;
};


/** \brief Contraction of two block tensors

    \ingroup libtensor_btod
 **/
template<size_t N, size_t M, size_t K>
class btod_contract2 :
    public additive_bto<N + M, btod_traits>,
    public timings< btod_contract2<N, M, K> > {

public:
    static const char *k_clazz; //!< Class name

private:
    enum {
        k_ordera = N + K, //!< Order of first argument (A)
        k_orderb = M + K, //!< Order of second argument (B)
        k_orderc = N + M, //!< Order of result (C)
        k_totidx = N + M + K, //!< Total number of indexes
        k_maxconn = 2 * k_totidx, //!< Index connections
    };

private:
    contraction2<N, M, K> m_contr; //!< Contraction
    block_tensor_i<k_ordera, double> &m_bta; //!< First argument (A)
    block_tensor_i<k_orderb, double> &m_btb; //!< Second argument (B)
    bto_contract2_sym<N, M, K, double> m_symc; //!< Symmetry of result (C)
    dimensions<k_ordera> m_bidimsa; //!< Block %index dims of A
    dimensions<k_orderb> m_bidimsb; //!< Block %index dims of B
    dimensions<k_orderc> m_bidimsc; //!< Block %index dims of the result
    assignment_schedule<k_orderc, double> m_sch; //!< Assignment schedule

public:
    //!    \name Construction and destruction
    //@{

    /** \brief Initializes the contraction operation
        \param contr Contraction.
        \param bta Block %tensor A (first argument).
        \param btb Block %tensor B (second argument).
    **/
    btod_contract2(const contraction2<N, M, K> &contr,
                   block_tensor_i<k_ordera, double> &bta,
                   block_tensor_i<k_orderb, double> &btb);

    /** \brief Virtual destructor
     **/
    virtual ~btod_contract2();

    //@}

    //!    \name Implementation of
    //      libtensor::direct_block_tensor_operation<N + M, double>
    //@{

    virtual const block_index_space<N + M> &get_bis() const;
    virtual const symmetry<N + M, double> &get_symmetry() const;
    virtual const assignment_schedule<N + M, double> &get_schedule() const;
    virtual void sync_on();
    virtual void sync_off();

    //@}

    virtual void perform(block_tensor_i<N + M, double> &btc);
    virtual void perform(block_tensor_i<N + M, double> &btc, const double &d);
    virtual void perform(bto_stream_i<N + M, btod_traits> &out);

    using additive_bto<N + M, btod_traits>::compute_block;
    virtual void compute_block(bool zero, dense_tensor_i<N + M, double> &blk,
        const index<N + M> &i, const tensor_transf<N + M, double> &tr,
        const double &c);

private:
    void perform_inner(
        block_tensor_i<N + K, double> &bta,
        block_tensor_i<M + K, double> &btb,
        block_tensor_i<N + M, double> &btc,
        double d,
        const std::vector<size_t> &blst);

    void make_schedule();

    void contract_block(
        block_tensor_i<N + K, double> &bta,
        const orbit_list<N + K, double> &ola,
        block_tensor_i<M + K, double> &btb,
        const orbit_list<M + K, double> &olb,
        const index<k_orderc> &idxc,
        dense_tensor_i<k_orderc, double> &blkc,
        const tensor_transf<k_orderc, double> &trc,
        bool zero, double c);

private:
    btod_contract2(const btod_contract2<N, M, K>&);
    btod_contract2<N, M, K> &operator=(const btod_contract2<N, M, K>&);

};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_CONTRACT2_H
