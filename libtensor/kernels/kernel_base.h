#ifndef LIBTENSOR_KERNEL_BASE_H
#define LIBTENSOR_KERNEL_BASE_H

#include <list>
#include "loop_list_node.h"
#include "loop_registers.h"

namespace libtensor {


/** \defgroup libtensor_kernels Multi-dimensional array math kernels
    \brief Math kernels to perform tensor operations
    \ingroup libtensor
 **/


/** \brief Base class for kernels
    \tparam N Number of input arrays.
    \tparam M Number of output arrays.

    \ingroup libtensor_kernels
 **/
template<size_t N, size_t M>
class kernel_base {
public:
    typedef std::list< loop_list_node<N, M> > list_t;
    typedef typename list_t::iterator iterator_t;

public:
    /** \brief Virtual destructor
     **/
    virtual ~kernel_base() { }

    /** \brief Returns the name of the kernel
     **/
    virtual const char *get_name() const = 0;

    /** \brief Runs the kernel
     **/
    virtual void run(const loop_registers<N, M> &r) = 0;

};


} // namespace libtensor

#endif // LIBTENSOR_KERNEL_BASE_H
