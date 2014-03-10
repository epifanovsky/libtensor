#ifndef LIBTENSOR_TO_EWMULT2_DIMS_H
#define LIBTENSOR_TO_EWMULT2_DIMS_H

#include <libtensor/core/noncopyable.h>
#include <libtensor/core/dimensions.h>

namespace libtensor {


/** \brief Computes the dimensions of the output of tod_ewmult2
    \tparam N Order of first tensor less the number of shared indices.
    \tparam M Order of second tensor less the number of shared indices.
    \tparam K Number of shared indices.

    \sa tod_ewmult2

    \ingroup libtensor_tod
 **/
template<size_t N, size_t M, size_t K>
class to_ewmult2_dims : public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

public:
    enum {
        NA = N + K,
        NB = M + K,
        NC = N + M + K
    };

private:
    dimensions<NC> m_dimsc; //!< Dimensions of the result

public:
    /** \brief Computes the dimensions
        \param dimsa Dimensions of first tensor (A).
        \param perma Permutation of A.
        \param dimsb Dimensions of second tensor (B).
        \param permb Permutation of B.
        \param permc Permutation of result.
     **/
    to_ewmult2_dims(
        const dimensions<NA> &dimsa, const permutation<NA> &perma,
        const dimensions<NB> &dimsb, const permutation<NB> &permb,
        const permutation<NC> &permc);

    /** \brief Returns the dimensions of the output
     **/
    const dimensions<NC> &get_dimsc() const {
        return m_dimsc;
    }

};


} // namespace libtensor

#endif // LIBTENSOR_TO_EWMULT2_DIMS_H
