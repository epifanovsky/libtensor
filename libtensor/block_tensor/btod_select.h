#ifndef LIBTENSOR_BTOD_SELECT_H
#define LIBTENSOR_BTOD_SELECT_H

#include <libtensor/core/noncopyable.h>
#include <libtensor/gen_block_tensor/gen_bto_select.h>
#include "btod_traits.h"

namespace libtensor {


/** \brief Selects a number of elements from a block %tensor
    \tparam N Tensor order.
    \tparam ComparePolicy Policy to select elements.

    The operation uses a block %tensor, a %symmetry and a compare policy to
    create an ordered list of block %tensor elements as (block %index,
    %index, value) data. The optional %symmetry is employed to determine the
    blocks from which elements can be selected. If it is not given, the
    internal symmetry of the block %tensor is used instead.

    Elements are selected exclusively from blocks which are unique, allowed,
    and non-zero within the given symmetry. If the symmetry by which the
    blocks are determined differs from the %symmetry of the block %tensor,
    the unique blocks within both symmetries might differ. Is this the case,
    a block present in the block %tensor might be transformed to yield the
    unique block within the symmetry before elements are selected.

    <b>Compare policy</b>

    The compare policy type determines the ordering of block %tensor elements
    by which they are selected. Any type used as compare policy needs to
    implement a function
    <code>
        bool operator( const double&, const double& )
    </code>
    which compares two block %tensor elements. If the function returns true,
    the first value is taken to be more optimal with respect to the compare
    policy.

    \ingroup libtensor_block_tensor_btod
 **/
template<size_t N, typename ComparePolicy=compare4absmin<double> >
class btod_select : public noncopyable {
public:
    static const char *k_clazz; //!< Class name

    typedef ComparePolicy compare_type;
    typedef gen_bto_select<N, btod_traits, compare_type> bto_select;
    typedef typename bto_select::block_tensor_element_type
            block_tensor_element_type;
    typedef std::list<block_tensor_element_type> list_type;

private:
    bto_select m_gbto; //!< Compare policy object to select entries

public:
    //! \name Constructor/destructor
    //@{

    /** \brief Constuctor without specific symmetry
         \param bt Block %tensor
        \param cmp Compare policy object (default: compare4absmin)
     **/
    btod_select(block_tensor_rd_i<N, double> &bt,
            compare_type cmp = compare_type());

    /** \brief Constuctor using symmetry
         \param bt Block %tensor
         \param sym Symmetry
        \param cmp Compare policy object (default: compare4absmin)
     **/
    btod_select(block_tensor_rd_i<N, double> &bt,
            const symmetry<N, double> &sym, compare_type cmp = compare_type());

    //@}

    /** \brief Performs the operation
        \param bt Block %tensor.
        \param li List of elements.
        \param n Maximum list size.
     **/
    void perform(list_type &li, size_t n);
};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_SELECT_H

#include "impl/btod_select_impl.h"
