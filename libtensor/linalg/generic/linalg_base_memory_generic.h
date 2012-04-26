#ifndef LIBTENSOR_LINALG_BASE_MEMORY_GENERIC_H
#define LIBTENSOR_LINALG_BASE_MEMORY_GENERIC_H

#include <cstdlib> // for size_t

namespace libtensor {


/** \brief Memory buffer allocation (generic)

    \ingroup libtensor_linalg
 **/
struct linalg_base_memory_generic {

    /** \brief Allocates a temporary array of doubles
        \param n Array length.
        \return Pointer to the array.
     **/
    static double *allocate(size_t n) {
        return new double[n];
    }

    /** \brief Deallocates a temporary array previously allocated
            using allocate(size_t)
        \param p Pointer to the array.
     **/
    static void deallocate(double *p) {
        delete [] p;
    }

};


} // namespace libtensor

#endif // LIBTENSOR_LINALG_BASE_MEMORY_GENERIC_H
