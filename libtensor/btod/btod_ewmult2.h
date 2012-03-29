#ifndef LIBTENSOR_BTOD_EWMULT2_H
#define LIBTENSOR_BTOD_EWMULT2_H

#include "additive_btod.h"

namespace libtensor {


/** \brief Generalized element-wise (Hadamard) product of two block tensors
    \tparam N Order of first argument (A) less the number of shared indexes.
    \tparam M Order of second argument (B) less the number of shared
        indexes.
    \tparam K Number of shared indexes.

    This operation computes the element-wise product of two block tensor.
    Refer to tod_ewmult2<N, M, K> for setup info.

    Both arguments and result must agree on their block index spaces,
    otherwise the constructor and perform() will raise
    bad_block_index_space.

    \sa tod_ewmult2, btod_contract2

    \ingroup libtensor_btod
 **/
template<size_t N, size_t M, size_t K>
class btod_ewmult2 :
    public additive_btod<N + M + K>,
    public timings< btod_ewmult2<N, M, K> > {

public:
    static const char *k_clazz; //!< Class name

public:
    enum {
        k_ordera = N + K, //!< Order of first argument (A)
        k_orderb = M + K, //!< Order of second argument (B)
        k_orderc = N + M + K //!< Order of result (C)
    };

private:
    block_tensor_i<k_ordera, double> &m_bta; //!< First argument (A)
    permutation<k_ordera> m_perma; //!< Permutation of first argument (A)
    block_tensor_i<k_orderb, double> &m_btb; //!< Second argument (B)
    permutation<k_orderb> m_permb; //!< Permutation of second argument (B)
    permutation<k_orderc> m_permc; //!< Permutation of result (C)
    double m_d; //!< Scaling coefficient
    block_index_space<k_orderc> m_bisc; //!< Block index space of result
    symmetry<k_orderc, double> m_symc; //!< Symmetry of result
    assignment_schedule<k_orderc, double> m_sch; //!< Assignment schedule

public:
    //!    \name Construction and destruction
    //@{

    /** \brief Initializes the operation
        \param bta First argument (A).
        \param btb Second argument (B).
        \param d Scaling coefficient.
     **/
    btod_ewmult2(block_tensor_i<k_ordera, double> &bta,
        block_tensor_i<k_orderb, double> &btb, double d = 1.0);

    /** \brief Initializes the operation
        \param bta First argument (A).
        \param perma Permutation of A.
        \param btb Second argument (B).
        \param permb Permutation of B.
        \param permc Permutation of result (C).
        \param recip Reciprocal flag.
        \param d Scaling coefficient.
     **/
    btod_ewmult2(block_tensor_i<k_ordera, double> &bta,
        const permutation<k_ordera> &perma,
        block_tensor_i<k_orderb, double> &btb,
        const permutation<k_orderb> &permb,
        const permutation<k_orderc> &permc, double d = 1.0);

    /** \brief Virtual destructor
     **/
    virtual ~btod_ewmult2();

    //@}


    //!    \name Implementation of
    //!        direct_block_tensor_operation<N + M + K, double>
    //@{

    virtual const block_index_space<N + M + K> &get_bis() const {
        return m_bisc;
    }

    virtual const symmetry<N + M + K, double> &get_symmetry() const {
        return m_symc;
    }

    virtual const assignment_schedule<N + M + K, double> &get_schedule()
        const {
        return m_sch;
    }

    virtual void sync_on();
    virtual void sync_off();

    using additive_btod<N + M + K>::perform;

    //@}

protected:
    virtual void compute_block(bool zero, dense_tensor_i<k_orderc, double> &blk,
        const index<k_orderc> &i, const tensor_transf<k_orderc, double> &tr,
        double c, cpu_pool &cpus);

private:
    /** \brief Computes the block index space of the result block tensor
     **/
    static block_index_space<N + M + K> make_bisc(
        const block_index_space<k_ordera> &bisa,
        const permutation<k_ordera> &perma,
        const block_index_space<k_orderb> &bisb,
        const permutation<k_orderb> &permb,
        const permutation<k_orderc> &permc);

    /** \brief Computes the symmetry of the result block tensor
     **/
    void make_symc();

    /** \brief Prepares the assignment schedule
     **/
    void make_schedule();

    /** \brief Computes the given block of the result
     **/
    void compute_block_impl(dense_tensor_i<k_orderc, double> &blk,
        const index<k_orderc> &bidx, const tensor_transf<k_orderc, double> &tr,
        bool zero, double d, cpu_pool &cpus);
};


} // namespace libtensor

#ifdef LIBTENSOR_INSTANTIATE_TEMPLATES

namespace libtensor {

    extern template class btod_ewmult2<0, 0, 1>;
    extern template class btod_ewmult2<0, 0, 2>;
    extern template class btod_ewmult2<0, 0, 3>;
    extern template class btod_ewmult2<0, 0, 4>;
    extern template class btod_ewmult2<0, 0, 5>;
    extern template class btod_ewmult2<0, 0, 6>;

    extern template class btod_ewmult2<0, 1, 1>;
    extern template class btod_ewmult2<0, 1, 2>;
    extern template class btod_ewmult2<0, 1, 3>;
    extern template class btod_ewmult2<0, 1, 4>;
    extern template class btod_ewmult2<0, 1, 5>;
    extern template class btod_ewmult2<1, 0, 1>;
    extern template class btod_ewmult2<1, 0, 2>;
    extern template class btod_ewmult2<1, 0, 3>;
    extern template class btod_ewmult2<1, 0, 4>;
    extern template class btod_ewmult2<1, 0, 5>;

    extern template class btod_ewmult2<0, 2, 1>;
    extern template class btod_ewmult2<0, 2, 2>;
    extern template class btod_ewmult2<0, 2, 3>;
    extern template class btod_ewmult2<0, 2, 4>;
    extern template class btod_ewmult2<1, 1, 1>;
    extern template class btod_ewmult2<1, 1, 2>;
    extern template class btod_ewmult2<1, 1, 3>;
    extern template class btod_ewmult2<1, 1, 4>;
    extern template class btod_ewmult2<2, 0, 1>;
    extern template class btod_ewmult2<2, 0, 2>;
    extern template class btod_ewmult2<2, 0, 3>;
    extern template class btod_ewmult2<2, 0, 4>;

    extern template class btod_ewmult2<0, 3, 1>;
    extern template class btod_ewmult2<0, 3, 2>;
    extern template class btod_ewmult2<0, 3, 3>;
    extern template class btod_ewmult2<1, 2, 1>;
    extern template class btod_ewmult2<1, 2, 2>;
    extern template class btod_ewmult2<1, 2, 3>;
    extern template class btod_ewmult2<2, 1, 1>;
    extern template class btod_ewmult2<2, 1, 2>;
    extern template class btod_ewmult2<2, 1, 3>;
    extern template class btod_ewmult2<3, 0, 1>;
    extern template class btod_ewmult2<3, 0, 2>;
    extern template class btod_ewmult2<3, 0, 3>;

    extern template class btod_ewmult2<0, 4, 1>;
    extern template class btod_ewmult2<0, 4, 2>;
    extern template class btod_ewmult2<1, 3, 1>;
    extern template class btod_ewmult2<1, 3, 2>;
    extern template class btod_ewmult2<2, 2, 1>;
    extern template class btod_ewmult2<2, 2, 2>;
    extern template class btod_ewmult2<3, 1, 1>;
    extern template class btod_ewmult2<3, 1, 2>;
    extern template class btod_ewmult2<4, 0, 1>;
    extern template class btod_ewmult2<4, 0, 2>;

    extern template class btod_ewmult2<0, 5, 1>;
    extern template class btod_ewmult2<1, 4, 1>;
    extern template class btod_ewmult2<2, 3, 1>;
    extern template class btod_ewmult2<3, 2, 1>;
    extern template class btod_ewmult2<4, 1, 1>;
    extern template class btod_ewmult2<5, 0, 1>;

} // namespace libtensor

#else // LIBTENSOR_INSTANTIATE_TEMPLATES
#include "btod_ewmult2_impl.h"
#endif // LIBTENSOR_INSTANTIATE_TEMPLATES

#endif // LIBTENSOR_BTOD_EWMULT2_H
