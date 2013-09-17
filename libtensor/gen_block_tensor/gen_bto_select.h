#ifndef LIBTENSOR_GEN_BTO_SELECT_H
#define LIBTENSOR_GEN_BTO_SELECT_H

#include <libtensor/defs.h>
#include <libtensor/core/block_tensor_element.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/core/symmetry.h>
#include <libtensor/core/transf_list.h>
#include "gen_block_tensor_i.h"


namespace libtensor {

/** \brief Selects a number of elements from a block %tensor
    \tparam N Tensor order.
    \tparam Traits Block tensor operation traits
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

    <b>Traits</b>

    The traits class has to provide definitions for
    - \c element_type -- Type of data elements
    - \c bti_traits -- Type of block tensor interface traits class
    - \c template to_select_type<N, ComparePolicy>::type -- Type of tensor
        operation to_select

    \ingroup libtensor_gen_bto
 **/
template<size_t N, typename Traits, typename ComparePolicy>
class gen_bto_select : public noncopyable {
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

    //! Type of compare policy
    typedef ComparePolicy compare_type;

    //! Type of block tensor element
    typedef block_tensor_element<N, element_type> block_tensor_element_type;

    //! Type of list for block tensor elements
    typedef std::list<block_tensor_element_type> list_type;

private:
    typedef typename Traits::template to_select_type<N, compare_type>::type
            to_select;
    typedef typename to_select::list_type to_list_type;
    typedef typename to_select::tensor_element_type tensor_element_type;

    gen_block_tensor_rd_i<N, bti_traits> &m_bt; //!< Block tensor to select data from
    symmetry<N, element_type> m_sym; //!< Symmetry imposed on block tensor
    compare_type m_cmp; //!< Compare policy object to select entries

public:
    //! \name Constructor/destructor
    //@{

    /** \brief Constuctor without specific symmetry
         \param bt Block %tensor
        \param cmp Compare policy object
     **/
    gen_bto_select(gen_block_tensor_rd_i<N, bti_traits> &bt,
            compare_type cmp = compare_type());

    /** \brief Constuctor using symmetry
         \param bt Block %tensor
         \param sym Symmetry
        \param cmp Compare policy object
     **/
    gen_bto_select(gen_block_tensor_rd_i<N, bti_traits> &bt,
            const symmetry<N, element_type> &sym,
            compare_type cmp = compare_type());

    //@}

    /** \brief Performs the operation
        \param bt Block %tensor.
        \param li List of elements.
        \param n Maximum list size.
     **/
    void perform(list_type &li, size_t n);


private:
    /** \brief Minimizes the list of tensor elements according to the list of
     		block transformations
        \param lst List of tensor elements
        \param trl List of block transformations
        \param dims Dimensions of current block
     **/
    void minimize_list(to_list_type &lst,
    		const transf_list<N, element_type> &trl,
    		const dimensions<N> &dims);

    void merge_lists(list_type &to, const index<N> &bidx,
            const to_list_type &from, size_t n);

};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_SELECT_H
