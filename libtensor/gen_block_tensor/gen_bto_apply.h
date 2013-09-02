#ifndef LIBTENSOR_GEN_BTO_APPLY_H
#define LIBTENSOR_GEN_BTO_APPLY_H

#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/core/symmetry.h>
#include <libtensor/core/tensor_transf.h>
#include "assignment_schedule.h"
#include "gen_block_stream_i.h"
#include "gen_block_tensor_i.h"

namespace libtensor {


/** \brief Applies a functor to each element of the input tensor.
    \tparam N Tensor order.
    \tparam Functor Functor to apply.
    \tparam Traits Traits class for this block tensor operation.
    \tparam Timed Class name to identify timer with.

    The operation transforms the input %tensor using \c tr1, then applies
    the functor to each element, and at last transforms the tensor using
    \c tr2. The functor class needs to have
    1. a proper copy constructor
      \code
          Functor(const Functor &f);
      \endcode
    2. an implementation of
      \code
          element_type Functor::operator()(const element_type &a);
      \endcode
    3. and implementations of
      \code
          bool Functor::keep_zero();
          const scalar_transf<element_type> &Functor::transf(bool arg);
      \endcode

    The latter two functions yield information about the symmetry of
    the functor:
    - keep_zero() -- Return true, if the functor maps zero to zero.
    - transf(bool) -- Return the two scalar transformations in
        \f$ f\left(\hat{T} x\right) = \hat{T}' f(x) \f$ (\f$\hat{T}\f$, if
        argument is true).

    The symmetry of the result tensor is determined by the symmetry operation
    so_apply. The use of this symmetry operation can result in the need to
    construct %tensor blocks from forbidden input %tensor blocks. Forbidden
    %tensor blocks are then treated as if they were zero.

    The traits class has to provide definitions for
    - \c element_type -- Type of data elements
    - \c bti_traits -- Type of block tensor interface traits class
    - \c template temp_block_type<N>::type -- Type of temporary tensor block
    - \c template temp_block_tensor_type<N>::type -- Type of temporary
            block tensor
    - \c template to_set_type<N>::type -- Type of tensor operation to_set
    - \c template to_copy_type<N>::type -- Type of tensor operation to_copy
    - \c template to_apply_type<N>::type -- Type of tensor operation to_apply

    \sa so_apply

    \ingroup libtensor_gen_bto
 **/
template<size_t N, typename Functor, typename Traits, typename Timed>
class gen_bto_apply : public timings<Timed>, public noncopyable {
public:
    static const char *k_clazz; //!< Class name

public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Block tensor interface traits
    typedef typename Traits::bti_traits bti_traits;

    //! Type of read-only block
    typedef typename bti_traits::template rd_block_type<N>::type rd_block_type;

    //! Type of write-only block
    typedef typename bti_traits::template wr_block_type<N>::type wr_block_type;

    //! Type of functor
    typedef Functor functor_type;

    typedef tensor_transf<N, element_type> tensor_transf_type;
    typedef scalar_transf<element_type> scalar_transf_type;

private:
    functor_type m_fn; //!< Functor to apply to each element
    gen_block_tensor_rd_i<N, bti_traits> &m_bta; //!< Source block %tensor
    scalar_transf_type m_tr1; //!< Scalar transformation applied before functor
    tensor_transf_type m_tr2; //!< Tensor transformation applied after functor
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
        \param tr1 Scalar transformation (applied before functor).
        \param tr1 Tensor transformation (applied after functor).
     **/
    gen_bto_apply(
            gen_block_tensor_rd_i<N, bti_traits> &bta,
            const functor_type &fn,
            const scalar_transf_type &tr1 = scalar_transf_type(),
            const tensor_transf_type &tr2 = tensor_transf_type());

    /** \brief Destructor
     **/
    virtual ~gen_bto_apply() { }

    //@}

    /** \brief Returns the block index space of the result
     **/
    const block_index_space<N> &get_bis() const {
        return m_bis;
    }

    /** \brief Returns the symmetry of the result
     **/
    const symmetry<N, element_type> &get_symmetry() const {
        return m_sym;
    }

    /** \brief Returns the list of canonical non-zero blocks of the result
     **/
    const assignment_schedule<N, element_type> &get_schedule() const {
        return m_sch;
    }

    /** \brief Computes and writes the blocks of the result to an output stream
        \param out Output stream.
     **/
    void perform(gen_block_stream_i<N, bti_traits> &out);

    /** \brief Computes one block of the result
        \param zero Zero target block first
        \param blkb Target block
        \param ib Index of target block
        \param trb Tensor transformation
     **/
    void compute_block(
            bool zero,
            const index<N> &ib,
            const tensor_transf_type &trb,
            wr_block_type &blkb);

    /** \brief Same as compute_block(), except it doesn't run a timer
     **/
    void compute_block_untimed(
            bool zero,
            const index<N> &ib,
            const tensor_transf_type &trb,
            wr_block_type &blkb);

private:
    /** \brief Forms the block %index space of the output or throws an
            exception if the input is incorrect.
     **/
    static block_index_space<N> mk_bis(const block_index_space<N> &bis,
            const permutation<N> &perm);

    /** \brief Sets up the assignment schedule for the operation.
     **/
    void make_schedule();
};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_APPLY_H
