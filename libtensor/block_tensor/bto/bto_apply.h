#ifndef LIBTENSOR_BTO_APPLY_H
#define LIBTENSOR_BTO_APPLY_H

#include <libtensor/timings.h>
#include <libtensor/block_tensor/bto/additive_bto.h>

namespace libtensor {


/** \brief Applies a functor, a permutation and a scaling coefficient
        to each element of the input tensor.
    \tparam N Tensor order.

    The operation scales and permutes the input %tensor and then applies
    the functor to each element. The functor class needs to have
    1. a proper copy constructor
      \code
          Functor(const Functor &f);
      \endcode
    2. an implementation of
      \code
          double Functor::operator()(const double &a);
      \endcode
    3. and implementations of
      \code
          bool Functor::keep_zero();
          bool Functor::transf(bool arg);
      \endcode

    The latter two functions yield information about the symmetry of
    the functor:
    - keep_zero() -- Return true, if the functor maps zero to zero.
    - transf(bool) -- Return the two scalar transformations in
        \f$ f\left(\hat{T}x\right) = \hat{T}' f(x) \f$ (\f$\hat{T}\f$, if
        argument is true).

    The symmetry of the result tensor is determined by the symmetry operation
    so_apply. The use of this symmetry operation can result in the need to
    construct %tensor blocks from forbidden input %tensor blocks. Forbidden
    %tensor blocks are then treated as if they where zero.

    \ingroup libtensor_btod
 **/
template<size_t N, typename Traits>
class bto_apply :
    public additive_bto<N, typename Traits::additive_bto_traits>,
    public timings< bto_apply<N, Traits> > {

public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_t;

    //! Type of block tensors
    typedef typename Traits::template block_tensor_type<N>::type
        block_tensor_t;

    //! Type of blocks of a block tensor
    typedef typename Traits::template block_type<N>::type block_t;

    //! Type of functor
    typedef typename Traits::functor_type functor_t;

    typedef tensor_transf<N, element_t> tensor_tr_t;

    typedef scalar_transf<element_t> scalar_tr_t;

public:
    static const char *k_clazz; //!< Class name

private:
    functor_t m_fn; //!< Functor to apply to each element
    block_tensor_t &m_bta; //!< Source block %tensor
    tensor_tr_t m_tr; //!< Tensor transformation
    block_index_space<N> m_bis; //!< Block %index space of output
    dimensions<N> m_bidims; //!< Block %index dimensions
    symmetry<N, element_t> m_sym; //!< Symmetry of output
    assignment_schedule<N, element_t> m_sch;

public:
    //!    \name Construction and destruction
    //@{

    /** \brief Initializes the element-wise operation
        \param bt Source block %tensor.
        \param f Functor instance.
        \param c Scaling coefficient.
     **/
    bto_apply(block_tensor_t &bta, const functor_t &fn,
            const scalar_tr_t &c = scalar_tr_t());

    /** \brief Initializes the permuted element-wise operation
        \param bt Source block %tensor.
        \param f Functor instance.
        \param p Permutation.
        \param c Scaling coefficient.
     **/
    bto_apply(block_tensor_t &bta, const functor_t &fn,
            const permutation<N> &p, const scalar_tr_t &c = scalar_tr_t());

    //@}

    //!    \name Implementation of
    //!        libtensor::direct_block_tensor_operation<N, double>
    //@{
    virtual const block_index_space<N> &get_bis() const {
        return m_bis;
    }

    virtual const symmetry<N, element_t> &get_symmetry() const {
        return m_sym;
    }

    using additive_bto<N, typename Traits::additive_bto_traits>::perform;

    virtual void sync_on();
    virtual void sync_off();

    //@}

    //!    \name Implementation of libtensor::btod_additive<N>
    //@{
    virtual const assignment_schedule<N, element_t> &get_schedule() const {
        return m_sch;
    }
    //@}

protected:
    virtual void compute_block(bool zero, block_t &blk,
            const index<N> &ib, const tensor_tr_t &tr,
            const scalar_tr_t &c, cpu_pool &cpus);

private:
    static block_index_space<N> mk_bis(const block_index_space<N> &bis,
            const permutation<N> &perm);
    void make_schedule();

private:
    bto_apply(const bto_apply<N, Traits>&);
    bto_apply<N, Traits> &operator=(const bto_apply<N, Traits>&);

};


} // namespace libtensor

#endif // LIBTENSOR_BTO_APPLY_H
