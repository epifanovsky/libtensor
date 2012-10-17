#ifndef LIBTENSOR_DIRECT_GEN_BLOCK_TENSOR_H
#define LIBTENSOR_DIRECT_GEN_BLOCK_TENSOR_H

#include <libutil/threads/mutex.h>
#include <libutil/threads/cond_map.h>
#include "direct_gen_block_tensor_base.h"


namespace libtensor {

/** \brief Direct generalized block %tensor
    \tparam N Tensor order.
    \tparam BtTraits Block %tensor traits.

    \ingroup libtensor_core
 **/
template<size_t N, typename BtTraits>
class direct_gen_block_tensor :
    public direct_gen_block_tensor_base<N, typename BtTraits::bti_traits> {

public:
    static const char *k_clazz; //!< Class name

public:
    //! Tensor element type
    typedef typename BtTraits::element_type element_type;

    //! Type of block %tensor interface traits
    typedef typename BtTraits::bti_traits bti_traits;

    //! Type of read-only block
    typedef typename bti_traits::template rd_block_type<N>::type rd_block_type;

    //! Type of read-write block
    typedef typename bti_traits::template wr_block_type<N>::type wr_block_type;

    //! Base class type
    typedef direct_gen_block_tensor_base<N, bti_traits> base_t;

    //! Type of block %tensor operation
    typedef typename base_t::operation_t operation_t;


private:
    dimensions<N> m_bidims; //!< Block %index dims
    libutil::mutex m_lock; //!< Mutex lock
    block_map<N, BtTraits> m_map; //!< Block map
    std::map<size_t, size_t> m_count; //!< Block count
    std::set<size_t> m_inprogress; //!< Computations in progress
    libutil::cond_map<size_t, size_t> m_cond; //!< Conditionals

public:
    //!    \name Construction and destruction
    //@{

    direct_gen_block_tensor(operation_t &op);
    virtual ~direct_gen_block_tensor() { }

    //@}

    using direct_gen_block_tensor_base<N, bti_traits>::get_bis;

protected:
    //!    \name Implementation of libtensor::gen_block_tensor_rd_i<N, bti_traits>
    //@{

    virtual const symmetry<N, element_type> &on_req_const_symmetry() {
        return get_op().get_symmetry();
    }
    virtual bool on_req_is_zero_block(const index<N> &idx);
    virtual rd_block_type &on_req_const_block(const index<N> &idx);
    virtual void on_ret_const_block(const index<N> &idx);

    //@}

    using direct_gen_block_tensor_base<N, bti_traits>::get_op;

private:
    //! \brief Performs calculation of the given block
    void perform(const index<N>& idx);
};


} // namespace libtensor

#endif // LIBTENSOR_DIRECT_BLOCK_TENSOR_H
