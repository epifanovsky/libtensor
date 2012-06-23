#ifndef LIBTENSOR_TO_CONTRACT2_DIMS_H
#define LIBTENSOR_TO_CONTRACT2_DIMS_H

#include <libtensor/core/dimensions.h>
#include <libtensor/tod/contraction2.h>

namespace libtensor {


/** \brief Computes the dimensions of the result of a tensor contraction

    \ingroup libtensor
 **/
template<size_t N, size_t M, size_t K>
class to_contract2_dims {
private:
    dimensions<N + M> m_dimsc; //!< Dimensions of result

public:
    /** \brief Computes the dimensions of C
        \param contr Contraction.
        \param dimsa Dimensions of A.
        \param dimsb Dimensions of B.
     **/
    to_contract2_dims(const contraction2<N, M, K> &contr,
        const dimensions<N + K> &dimsa, const dimensions<M + K> &dimsb) :
        m_dimsc(make_dimsc(contr, dimsa, dimsb))
    { }

    /** \brief Returns the dimensions of C
     **/
    const dimensions<N + M> &get_dimsc() const {
        return m_dimsc;
    }

private:
    static dimensions<N + M> make_dimsc(const contraction2<N, M, K> &contr,
        const dimensions<N + K> &dimsa, const dimensions<M + K> &dimsb);

private:
    /** \brief Private copy constructor
     **/
    to_contract2_dims(const to_contract2_dims&);

};


} // namespace libtensor

#endif // LIBTENSOR_TO_CONTRACT2_DIMS_H
