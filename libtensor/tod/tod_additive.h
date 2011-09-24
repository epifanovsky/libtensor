#ifndef LIBTENSOR_TOD_ADDITIVE_H
#define LIBTENSOR_TOD_ADDITIVE_H

#include "../core/tensor_i.h"
#include "../mp/cpu_pool.h"

namespace libtensor {


/** \brief Additive tensor operation (double)

    A %tensor operation that operates on the double precision element data
    type should implement this interface if its sole result is a %tensor
    and it can add it an existing %tensor without allocating a buffer.

    \ingroup libtensor_tod
 **/
template<size_t N>
class tod_additive {
public:
    /** \brief Virtual destructor
     **/
    virtual ~tod_additive() {

    }

    /**	\brief Prefetches the operands
     **/
    virtual void prefetch() = 0;

    /**	\brief Performs the operation and writes the result to a %tensor
        \param cpus CPU pool.
        \param zero Whether to zero out before adding the result.
        \param c Coefficient.
        \param t Output tensor.
     **/
    virtual void perform(cpu_pool &cpus, bool zero, double c,
        tensor_i<N, double> &t) = 0;

    virtual void perform(tensor_i<N, double> &t, double c) { }
    virtual void perform(tensor_i<N, double> &t) { }

};


} // namespace libtensor

#endif // LIBTENSOR_TOD_ADDITIVE_H
