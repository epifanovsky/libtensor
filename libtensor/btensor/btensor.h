#ifndef LIBTENSOR_BTENSOR_H
#define LIBTENSOR_BTENSOR_H

#include <memory>
#include <libtensor/core/allocator.h>
#include <libtensor/core/immutable.h>
#include <libtensor/core/block_tensor.h>
#include "bispace.h"
#include "btensor_i.h"

namespace libtensor {


/** \brief Block tensor proxy for tensor expressions (base class)

    \ingroup libtensor_expr
 **/
template<size_t N, typename T, typename Alloc>
class btensor_base {
private:
    block_tensor<N, T, Alloc> m_bt; //!< Block tensor

public:
    btensor_base(const block_index_space<N> &bis) : m_bt(bis) { }

protected:
    block_tensor<N, T, Alloc> &get_bt() {
        return m_bt;
    }

};


/** \brief Block tensor proxy for tensor expressions

    \ingroup libtensor_expr
 **/
template<size_t N, typename T = double, typename Alloc = allocator<double> >
class btensor :
    public btensor_base<N, T, Alloc>,
    public btensor_i<N, T>,
    public immutable {

public:
    //!	\name Construction and destruction
    //@{

    /**	\brief Constructs the tensor in the given space
        \param bis Block index space
     **/
    btensor(const bispace<N> &bis) :
        btensor_base<N, T, Alloc>(bis.get_bis()),
        btensor_i<N, T>(btensor_base<N, T, Alloc>::get_bt()) { }

    /** \brief Constructs the tensor in the given space
        \param bis Block index space
     **/
    btensor(const block_index_space<N> &bis) :
        btensor_base<N, T, Alloc>(bis),
        btensor_i<N, T>(btensor_base<N, T, Alloc>::get_bt()) { }

    /**	\brief Virtual destructor
     **/
    virtual ~btensor() { }

    //@}

protected:
    //!	\name Implementation of libtensor::immutable
    //@{

    virtual void on_set_immutable() {
        btensor_base<N, T, Alloc>::get_bt().set_immutable();
    }

    //@}

};


} // namespace libtensor

#endif // LIBTENSOR_BTENSOR_H

