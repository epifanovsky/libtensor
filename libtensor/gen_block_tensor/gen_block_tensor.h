#ifndef LIBTENSOR_GEN_BLOCK_TENSOR_H
#define LIBTENSOR_GEN_BLOCK_TENSOR_H

#include <libutil/threads/mutex.h>
#include <libtensor/core/block_index_space.h>
#include <libtensor/core/immutable.h>
#include <libtensor/core/noncopyable.h>
#include "block_map.h"
#include "gen_block_tensor_i.h"

namespace libtensor {


/** \brief General block tensor
    \tparam N Tensor order.
    \tparam BtTraits Block tensor traits.

	This class is an implementation of the block %tensor concept for arbitrary
	types of %tensor blocks. It stores the list of unique, non-zero blocks of
	a block %tensor and provides methods to access and modify these %tensor
	blocks. It does not allocate or deallocate any data for %tensor blocks
	but assumes that the %tensor block class does this by itself.

	<b>Block %tensor traits</b>

	The template parameter \c BtTraits has to provide the information on
	the type of the %tensor blocks, as well as the data element type.
	Any class / structure to be used as \c BtTraits has to define the following
	types
	- \c element_type -- Type of the data elements
	- \c template block_type<N>::type -- Type of %tensor blocks
	- \c template block_factory_type<N>::type -- Factory type to create %tensor
		blocks
	- \c bti_traits -- Traits for the block %tensor interface
		(\sa gen_block_tensor_i).

	<b>Block %tensor structure</b>

	The block structure of the block %tensor is described by the
	block %index space object passed to the constructor. In addition,
	symmetry can be imposed on the block %tensor by adding or removing
	symmetry elements to or from the internal symmetry object.
	Modification of the symmetry object has to proceed via control objects
	which provide controlled access to the interface methods. Changes to
	the symmetry object also change the internal structure of the block
	%tensor. Thus, the symmetry should be set up before data is written
	to any of the %tensor blocks in order to avoid inconsistencies.

	<b>Block storage</b>

	Any block %tensor is initialized empty with no allocation taking place.
	Only when blocks are requested for writing for the first time, they are
	created as objects of the %tensor block type provided via the traits.
	The %tensor block class is expected to take care of the allocation of data
	elements. Overall only non-zero blocks which are unique with respect to
	symmetry are stored.

	<b>Operations on block %tensor</b>

	No mathematical operations on block tensors are implemented by this class.
	Instead access to the %tensor blocks is provided via interface methods, so
	that any operation of arbitrary complexity can be implemented in a separate
	class.

    \ingroup libtensor_gen_block_tensor
 **/
template<size_t N, typename BtTraits>
class gen_block_tensor :
    virtual public gen_block_tensor_i<N, typename BtTraits::bti_traits>,
    public immutable,
    public noncopyable {

public:
    static const char k_clazz[]; //!< Class name

public:
    typedef typename BtTraits::element_type element_type;
    typedef typename BtTraits::bti_traits bti_traits;
    typedef typename bti_traits::template rd_block_type<N>::type rd_block_type;
    typedef typename bti_traits::template wr_block_type<N>::type wr_block_type;
    typedef typename BtTraits::template block_type<N>::type block_type;
    typedef symmetry<N, element_type> symmetry_type;

public:
    block_index_space<N> m_bis; //!< Block index space
    dimensions<N> m_bidims; //!< Block index dimensions
    symmetry<N, element_type> m_symmetry; //!< Block tensor symmetry
    block_map<N, BtTraits> m_map; //!< Block map
    libutil::mutex m_lock; //!< Read-write lock

public:
    //!    \name Construction and destruction
    //@{
    gen_block_tensor(const block_index_space<N> &bis);
    virtual ~gen_block_tensor();
    //@}

    //!    \name Implementation of libtensor::gen_block_tensor_i<N, bti_traits>
    //@{
    virtual const block_index_space<N> &get_bis() const;
    //@}

protected:
    //!    \name Implementation of libtensor::gen_block_tensor_i<N, bti_traits>
    //@{
    virtual const symmetry_type &on_req_const_symmetry();
    virtual symmetry_type &on_req_symmetry();
    virtual rd_block_type &on_req_const_block(const index<N> &idx);
    virtual void on_ret_const_block(const index<N> &idx);
    virtual wr_block_type &on_req_block(const index<N> &idx);
    virtual void on_ret_block(const index<N> &idx);
    virtual bool on_req_is_zero_block(const index<N> &idx);
    virtual void on_req_nonzero_blocks(std::vector<size_t> &nzlst);
    virtual void on_req_zero_block(const index<N> &idx);
    virtual void on_req_zero_all_blocks();
    //@}

    //!    \name Implementation of libtensor::immutable
    //@{
    virtual void on_set_immutable();
    //@}

private:
    bool check_canonical_block(const index<N> &idx);
    block_type &get_block(const index<N> &idx, bool create);

};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BLOCK_TENSOR_H
