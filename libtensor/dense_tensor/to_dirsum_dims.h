#ifndef LIBTENSOR_TO_DIRSUM_DIMS_H
#define LIBTENSOR_TO_DIRSUM_DIMS_H

#include <libtensor/core/noncopyable.h>
#include <libtensor/core/dimensions.h>
#include <libtensor/core/mask.h>

namespace libtensor {


/** \brief Computes the dimensions of the output of tod_dirsum
    \tparam N Order of first tensor.
    \tparam M Order of second tensor.

    \sa tod_dirsum

    \ingroup libtensor_tod
 **/
template<size_t N, size_t M>
class to_dirsum_dims : public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

public:
    enum {
        NA = N,
        NB = M,
        NC = N + M
    };

private:
    dimensions<NC> m_dimsc; //!< Dimensions of the result

public:
    /** \brief Computes the dimensions
        \param dimsa Dimensions of first tensor (A).
        \param dimsb Dimensions of second tensor (B).
        \param permc Permutation of result.
     **/
    to_dirsum_dims(
        const dimensions<NA> &dimsa,
        const dimensions<NB> &dimsb,
        const permutation<NC> &permc);

    /** \brief Returns the dimensions of the output
     **/
    const dimensions<NC> &get_dimsc() const {
        return m_dimsc;
    }

};


} // namespace libtensor

#endif // LIBTENSOR_TO_DIRSUM_DIMS_H
