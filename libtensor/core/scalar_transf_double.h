#ifndef LIBTENSOR_SCALAR_TRANSF_DOUBLE_H
#define LIBTENSOR_SCALAR_TRANSF_DOUBLE_H

#include "scalar_transf.h"


namespace libtensor {


/** \brief Specialization of scalar_transf<T> for T == double
 **/
template<>
class scalar_transf<double> {
private:
    double m_coeff; //!< Coefficient

public:
    //! \name Constructors
    //@{

    /** \brief Default constructor
        \param coeff Scaling coefficient (default: 1.0)
     **/
    explicit scalar_transf(double coeff = 1.0) : m_coeff(coeff) { }

    /** \brief Copy constructor
     **/
    scalar_transf(const scalar_transf<double> &tr) : m_coeff(tr.m_coeff) { }

    /** \brief Assigment operator
     **/
    scalar_transf<double> &operator=(const scalar_transf<double> &tr) {
        m_coeff = tr.m_coeff;
        return *this;
    }

    //@}


    //! \name Manipulating functions
    //@ {

    void reset() { m_coeff = 1.0; }

    scalar_transf<double> &transform(const scalar_transf<double> &tr);

    scalar_transf<double> &invert();

    void apply(double &el) const { el *= m_coeff; }

    //@}

    //! \name Functions specific for T = double
    //@{

    /** \brief Scale coefficient by c
     **/
    void scale(double c) { m_coeff *= c; }


    /** \brief Returns the coefficient
     **/
    const double& get_coeff() const { return m_coeff; }

    //@}

    //! Comparison functions and operators
    //@{

    /** \brief True, if the transformation leaves the elements unchanged
     **/
    bool is_identity() const { return m_coeff == 1.0; }

    /** \brief True if all elements are mapped to zero.
     **/
    bool is_zero() const { return m_coeff == 0.0; }

    /** \brief equal comparison
     **/
    bool operator==(const scalar_transf<double>& tr) const {
        return (m_coeff==tr.m_coeff);
    }

    /** \brief Unequal comparison
     **/
    bool operator!=(const scalar_transf<double>& tr) const {
        return (!operator==(tr));
    }

    //@}
};


/** \brief Specialization of scalar_transf_sum<T> for T == double
 **/
template<>
class scalar_transf_sum<double> {
private:
    double m_coeff; //!< Coefficient

public:
    /** \brief Default constructor
        \param coeff Scaling coefficient (default: 1.0)
     **/
    scalar_transf_sum() : m_coeff(0.0) { }

    /** \brief Add scalar transformation to sum
     */
    void add(const scalar_transf<double> &tr) {
        m_coeff += tr.get_coeff();
    }

    /** \brief Return the result transformation
     **/
    scalar_transf<double> get_transf() const {
        return scalar_transf<double>(m_coeff);
    }

    /** \brief Apply sum to element
     **/
    void apply(double &el) const { el *= m_coeff; }

    /** \brief True, if the transformation leaves the elements unchanged
     **/
    bool is_identity() const { return m_coeff == 1.0; }

    /** \brief True if all elements are mapped to zero.
     **/
    bool is_zero() const { return m_coeff == 0.0; }
};


inline
scalar_transf<double> &scalar_transf<double>::transform(
        const scalar_transf<double> &tr) {

    m_coeff *= tr.m_coeff; return *this;
}


inline
scalar_transf<double> &scalar_transf<double>::invert() {

    m_coeff = (m_coeff == 0.0 ? 0.0 : 1.0/m_coeff);
    return *this;
}


inline std::ostream &operator<<(std::ostream &os,
        const scalar_transf<double> &tr) {
    os << tr.get_coeff();
    return os;
}


} // namespace libtensor


#endif // LIBTENSOR_SCALAR_TRANSF_DOUBLE_H


#ifndef LIBTENSOR_SCALAR_TRANSF_FLOAT_H
#define LIBTENSOR_SCALAR_TRANSF_FLOAT_H

#include "scalar_transf.h"


namespace libtensor {


/** \brief Specialization of scalar_transf<T> for T == float
 **/
template<>
class scalar_transf<float> {
private:
    float m_coeff; //!< Coefficient

public:
    //! \name Constructors
    //@{

    /** \brief Default constructor
        \param coeff Scaling coefficient (default: 1.0)
     **/
    explicit scalar_transf(float coeff = 1.0) : m_coeff(coeff) { }

    /** \brief Copy constructor
     **/
    scalar_transf(const scalar_transf<float> &tr) : m_coeff(tr.m_coeff) { }

    /** \brief Assigment operator
     **/
    scalar_transf<float> &operator=(const scalar_transf<float> &tr) {
        m_coeff = tr.m_coeff;
        return *this;
    }

    //@}


    //! \name Manipulating functions
    //@ {

    void reset() { m_coeff = 1.0; }

    scalar_transf<float> &transform(const scalar_transf<float> &tr);

    scalar_transf<float> &invert();

    void apply(float &el) const { el *= m_coeff; }

    //@}

    //! \name Functions specific for T = float
    //@{

    /** \brief Scale coefficient by c
     **/
    void scale(float c) { m_coeff *= c; }


    /** \brief Returns the coefficient
     **/
    const float& get_coeff() const { return m_coeff; }

    //@}

    //! Comparison functions and operators
    //@{

    /** \brief True, if the transformation leaves the elements unchanged
     **/
    bool is_identity() const { return m_coeff == 1.0; }

    /** \brief True if all elements are mapped to zero.
     **/
    bool is_zero() const { return m_coeff == 0.0; }

    /** \brief equal comparison
     **/
    bool operator==(const scalar_transf<float>& tr) const {
        return (m_coeff==tr.m_coeff);
    }

    /** \brief Unequal comparison
     **/
    bool operator!=(const scalar_transf<float>& tr) const {
        return (!operator==(tr));
    }

    //@}
};


/** \brief Specialization of scalar_transf_sum<T> for T == float
 **/
template<>
class scalar_transf_sum<float> {
private:
    float m_coeff; //!< Coefficient

public:
    /** \brief Default constructor
        \param coeff Scaling coefficient (default: 1.0)
     **/
    scalar_transf_sum() : m_coeff(0.0) { }

    /** \brief Add scalar transformation to sum
     */
    void add(const scalar_transf<float> &tr) {
        m_coeff += tr.get_coeff();
    }

    /** \brief Return the result transformation
     **/
    scalar_transf<float> get_transf() const {
        return scalar_transf<float>(m_coeff);
    }

    /** \brief Apply sum to element
     **/
    void apply(float &el) const { el *= m_coeff; }

    /** \brief True, if the transformation leaves the elements unchanged
     **/
    bool is_identity() const { return m_coeff == 1.0; }

    /** \brief True if all elements are mapped to zero.
     **/
    bool is_zero() const { return m_coeff == 0.0; }
};


inline
scalar_transf<float> &scalar_transf<float>::transform(
        const scalar_transf<float> &tr) {

    m_coeff *= tr.m_coeff; return *this;
}


inline
scalar_transf<float> &scalar_transf<float>::invert() {

    m_coeff = (m_coeff == 0.0 ? 0.0 : 1.0/m_coeff);
    return *this;
}


inline std::ostream &operator<<(std::ostream &os,
        const scalar_transf<float> &tr) {
    os << tr.get_coeff();
    return os;
}


} // namespace libtensor


#endif // LIBTENSOR_SCALAR_TRANSF_FLOAT_H
