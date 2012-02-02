#ifndef LIBTENSOR_BTENSOR_I_H
#define LIBTENSOR_BTENSOR_I_H

#include <libtensor/core/block_tensor_ctrl.h>
#include <libtensor/expr/anytensor.h>
#include <libtensor/expr/expression_dispatcher.h>
#include "btensor_renderer.h"

namespace libtensor {


/** \brief Block tensor proxy for tensor expressions

    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
class btensor_i :
    public block_tensor_i<N, T>,
    public anytensor<N, T> {

public:
    static const char *k_tensor_type; //!< Tensor type

private:
    block_tensor_i<N, T> &m_bt; //!< Block tensor
    block_tensor_ctrl<N, T> m_ctrl; //!< Block tensor control

public:
    //! \name Construction and destruction
    //@{

    /** \brief Initializes the wrapper
        \param bis Block index space
     **/
    btensor_i(block_tensor_i<N, T> &bt) :
        anytensor<N, T>(bt), m_bt(bt), m_ctrl(m_bt) {
        expression_dispatcher<N, T>::get_instance().register_renderer(
            k_tensor_type, btensor_renderer<N, T>());
    }

    /** \brief Virtual destructor
     **/
    virtual ~btensor_i() { }

    //@}

    //! \name Implementation of block_tensor_i<N, T>
    //@{

    virtual const block_index_space<N> &get_bis() const {
        return m_bt.get_bis();
    }

    //@}

    //! \name Implementation of anytensor<N, T>
    //@{

    virtual const char *get_tensor_type() const {
        return k_tensor_type;
    }

    //@}

protected:
    //! \name Implementation of libtensor::block_tensor_i<N,T>
    //@{

    virtual symmetry<N, T> &on_req_symmetry() throw(exception) {
        return m_ctrl.req_symmetry();
    }

    virtual const symmetry<N, T> &on_req_const_symmetry() throw(exception) {
        return m_ctrl.req_const_symmetry();
    }

    virtual dense_tensor_i<N, T> &on_req_block(const index<N> &idx)
        throw(exception) {
        return m_ctrl.req_block(idx);
    }

    virtual void on_ret_block(const index<N> &idx) throw(exception) {
        m_ctrl.ret_block(idx);
    }

    virtual dense_tensor_i<N, T> &on_req_aux_block(const index<N> &idx)
        throw(exception) {
        return m_ctrl.req_aux_block(idx);
    }

    virtual void on_ret_aux_block(const index<N> &idx) throw(exception) {
        m_ctrl.ret_aux_block(idx);
    }

    virtual bool on_req_is_zero_block(const index<N> &idx) throw(exception) {
        return m_ctrl.req_is_zero_block(idx);
    }

    virtual void on_req_zero_block(const index<N> &idx) throw(exception) {
        m_ctrl.req_zero_block(idx);
    }

    virtual void on_req_zero_all_blocks() throw(exception) {
        m_ctrl.req_zero_all_blocks();
    }

    virtual void on_req_sync_on() throw(exception) {
        m_ctrl.req_sync_on();
    }

    virtual void on_req_sync_off() throw(exception) {
        m_ctrl.req_sync_off();
    }

    //@}

};


template<size_t N, typename T>
const char *btensor_i<N, T>::k_tensor_type = "btensor";


} // namespace libtensor

#endif // LIBTENSOR_BTENSOR_I_H
