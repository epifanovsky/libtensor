#ifndef LIBTENSOR_SCALAR_TRANSF_COMPLEX_DOUBLE_H
#define LIBTENSOR_SCALAR_TRANSF_COMPLEX_DOUBLE_H

#include <complex>
#include "scalar_transf.h"


namespace libtensor {


/** \brief Specialization of scalar_transf<T> for T == std::complex<double>
 **/
template<>
class scalar_transf< std::complex<double> > {
private:
    std::complex<double> m_coeff; //!< Coefficient

public:
    //! \name Constructors
    //@{

    /** \brief Default constructor
        \param coeff Scaling coefficient (default: 1.0)
     **/
    explicit scalar_transf(std::complex<double> coeff =
            std::complex<double>(1.0, 0.0)) :
        m_coeff(coeff)
    { }

    /** \brief Copy constructor
     **/
    scalar_transf(const scalar_transf< std::complex<double> > &tr) :
        m_coeff(tr.m_coeff)
    { }

    //@}


    //! \name Manipulating functions
    //@ {

    void reset() { m_coeff = std::complex<double>(1.0, 0.0); }

    scalar_transf< std::complex<double> > &transform(
        const scalar_transf< std::complex<double> > &tr);

    scalar_transf< std::complex<double> > &invert();

    void apply(std::complex<double> &el) const { el *= m_coeff; }

    //@}

    //! \name Functions specific for T = std::complex<double>
    //@{

    /** \brief Scale coefficient by c
     **/
    void scale(const std::complex<double> &c) { m_coeff *= c; }


    /** \brief Returns the coefficient
     **/
    const std::complex<double>& get_coeff() const { return m_coeff; }

    //@}

    //! Comparison functions and operators
    //@{

    /** \brief True, if the transformation leaves the elements unchanged
     **/
    bool is_identity() const {
        return m_coeff == std::complex<double>(1.0, 0.0);
    }

    /** \brief True if all elements are mapped to zero.
     **/
    bool is_zero() const {
        return m_coeff == std::complex<double>(0.0, 0.0);
    }

    /** \brief equal comparison
     **/
    bool operator==(const scalar_transf< std::complex<double> >& tr) const {

        return (m_coeff == tr.m_coeff);
    }

    /** \brief Unequal comparison
     **/
    bool operator!=(const scalar_transf< std::complex<double> >& tr) const {
        return (!operator==(tr));
    }

    //@}
};


/** \brief Specialization of scalar_transf_sum<T> for T == std::complex<double>
 **/
template<>
class scalar_transf_sum< std::complex<double> > {
private:
    std::complex<double> m_coeff; //!< Coefficient

public:
    /** \brief Default constructor
        \param coeff Scaling coefficient (default: 1.0)
     **/
    scalar_transf_sum() : m_coeff(0.0) { }

    /** \brief Add scalar transformation to sum
     */
    void add(const scalar_transf< std::complex<double> > &tr) {
        m_coeff += tr.get_coeff();
    }

    /** \brief Return the result transformation
     **/
    scalar_transf< std::complex<double> > get_transf() const {
        return scalar_transf< std::complex<double> >(m_coeff);
    }

    /** \brief Apply sum to element
     **/
    void apply(std::complex<double> &el) const { el *= m_coeff; }

    /** \brief True, if the transformation leaves the elements unchanged
     **/
    bool is_identity() const {
        return m_coeff == std::complex<double>(1.0, 0.0);
    }

    /** \brief True if all elements are mapped to zero.
     **/
    bool is_zero() const {
        return m_coeff == std::complex<double>(0.0, 0.0);
    }

};


inline
scalar_transf< std::complex<double> > &scalar_transf<std::complex<double> >::
        transform(const scalar_transf< std::complex<double> > &tr) {

    m_coeff *= tr.m_coeff; return *this;
}


inline
scalar_transf< std::complex<double> > &scalar_transf< std::complex<double> >::
    invert() {

    m_coeff = is_zero() ? std::complex<double>(0.0, 0.0) : 1.0 / m_coeff;
    return *this;
}


inline std::ostream &operator<<(std::ostream &os,
        const scalar_transf< std::complex<double> > &tr) {
    os << tr.get_coeff();
    return os;
}


} // namespace libtensor


#endif // LIBTENSOR_SCALAR_TRANSF_COMPLEX_DOUBLE_H
