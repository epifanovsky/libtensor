#ifndef LIBTENSOR_BTO_APPLY_H
#define LIBTENSOR_BTO_APPLY_H

#include <libtensor/timings.h>
#include <libtensor/block_tensor/bto/additive_bto.h>

namespace libtensor {


/** \brief Applies a functor, a permutation and a scaling coefficient
        to each element of the input tensor.
    \tparam N Tensor order.

    The operation transforms the input %tensor using \c tr1, then applies
    the functor to each element, and at last transforms the tensor using
    \c tr2. The functor class needs to have
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
template<size_t N, typename Functor, typename Traits>
class bto_apply :
    public additive_bto<N, Traits>,
    public timings< bto_apply<N, Functor, Traits> > {

public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Type of block tensors
    typedef typename Traits::template block_tensor_type<N>::type
        block_tensor_type;

    //! Type of blocks of a block tensor
    typedef typename Traits::template block_type<N>::type block_type;

    //! Type of functor
    typedef Functor functor_type;

    typedef scalar_transf<element_type> scalar_transf_type;
    typedef tensor_transf<N, element_type> tensor_transf_type;

public:
    static const char *k_clazz; //!< Class name

private:
    functor_type m_fn; //!< Functor to apply to each element
    block_tensor_type &m_bta; //!< Source block %tensor
    scalar_transf_type m_tr1; //!< Scalar transformation before
    tensor_transf_type m_tr2; //!< Tensor transformation after
    block_index_space<N> m_bis; //!< Block %index space of output
    dimensions<N> m_bidims; //!< Block %index dimensions
    symmetry<N, element_type> m_sym; //!< Symmetry of output
    assignment_schedule<N, element_type> m_sch;

public:
    //!    \name Construction and destruction
    //@{

    /** \brief Initializes the element-wise operation
        \param bt Source block %tensor.
        \param fn Functor instance.
        \param tr1 Scalar transformation applied before functor.
        \param tr2 Tensor transformation applied after functor.
     **/
    bto_apply(block_tensor_type &bta, const functor_type &fn,
            const scalar_transf_type &tr1 = scalar_transf_type(),
            const tensor_transf_type &tr2 = tensor_transf_type());

    /** \brief Initializes the permuted element-wise operation
        \param bt Source block %tensor.
        \param fn Functor instance.
        \param p Permutation (apply before).
        \param c Scaling coefficient (apply before).
     **/
    bto_apply(block_tensor_type &bta,
            const functor_type &fn, const permutation<N> &p,
            const scalar_transf_type &c = scalar_transf_type());

    /** \brief Destructor
     **/
    virtual ~bto_apply() { }

    //@}

    //!    \name Implementation of
    //!        libtensor::direct_block_tensor_operation<N, double>
    //@{
    virtual const block_index_space<N> &get_bis() const {
        return m_bis;
    }

    virtual const symmetry<N, element_type> &get_symmetry() const {
        return m_sym;
    }

    virtual void sync_on();
    virtual void sync_off();

    virtual void perform(bto_stream_i<N, Traits> &out);
    virtual void perform(block_tensor_type &btb);
    virtual void perform(block_tensor_type &btb, const element_type &c);

    //@}

    //!    \name Implementation of libtensor::btod_additive<N>
    //@{
    virtual const assignment_schedule<N, element_type> &get_schedule() const {
        return m_sch;
    }
    //@}

    virtual void compute_block(bool zero, block_type &blk, const index<N> &ib,
            const tensor_transf_type &tr, const element_type &c);

private:
    static block_index_space<N> mk_bis(const block_index_space<N> &bis,
            const permutation<N> &perm);
    void make_schedule();

private:
    bto_apply(const bto_apply&);
    bto_apply &operator=(const bto_apply&);

};


} // namespace libtensor

#endif // LIBTENSOR_BTO_APPLY_H
