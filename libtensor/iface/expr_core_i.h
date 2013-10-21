#ifndef LIBTENSOR_IFACE_EXPR_CORE_I_H
#define LIBTENSOR_IFACE_EXPR_CORE_I_H

namespace libtensor {
namespace iface {

template<size_t N, typename T>
class expr_core_ptr;

/** \brief Expression core base class

    TODO: Extend interface: costs, etc

    \ingroup libtensor_iface
 **/
template<size_t N, typename T>
class expr_core_i {
public:
    /** \brief Virtual destructor
     **/
    virtual ~expr_core_i() { }

    /** \brief Returns type of the expression core
     **/
    virtual const std::string &get_type() const = 0;
};


/** \brief Reference counted pointer to the expression core

    \ingroup libtensor_iface
 **/
template<size_t N, typename T>
class expr_core_ptr {
private:
    struct counter {
        expr_core_i<N, T> *m_core; //!< Expression core
        size_t m_count;
        counter(expr_core_i<N, T> *core);
    } *m_expr; //!< Expression core pointer

public:
    /** \brief Acquire expression core
     **/
    expr_core_ptr(expr_core_i<N, T> *core = 0);

    /** \brief Copy constructor
     **/
    expr_core_ptr(const expr_core_ptr<N, T> &ptr);

    /** \brief Destructor
     **/
    ~expr_core_ptr();

    /** \brief Assignment operator
     **/
    expr_core_ptr<N, T> &operator=(const expr_core_ptr<N, T> &ptr);

    /** \brief Return expression core
     **/
    expr_core_i<N, T> &get_core();

    /** \brief Return expression core (const version)
     **/
    const expr_core_i<N, T> &get_core() const;

    /** \brief Return expression core
     **/
    template<typename Core>
    Core &get_core();

    /** \brief Return expression core (const version)
     **/
    template<typename Core>
    const Core &get_core() const;

private:
    void acquire(counter *expr);
    void release();
};


template<size_t N, typename T>
inline expr_core_ptr<N, T>::expr_core_ptr(expr_core_i<N, T> *core) {

    if(core) m_expr = new counter(core);
    else m_expr = NULL;

}


template<size_t N, typename T>
inline expr_core_ptr<N, T>::expr_core_ptr(const expr_core_ptr<N, T> &ptr) {

    acquire(ptr.m_expr);
}


template<size_t N, typename T>
inline expr_core_ptr<N, T>::~expr_core_ptr() {

    release();
}


template<size_t N, typename T>
inline expr_core_ptr<N, T> &expr_core_ptr<N, T>::operator=(
        const expr_core_ptr<N, T> &ptr) {

    if(this != &ptr) {
        release(); acquire(ptr.m_expr);
    }
    return *this;
}


template<size_t N, typename T>
void expr_core_ptr<N, T>::acquire(counter *expr) {

    m_expr = expr;
    if(m_expr) m_expr->m_count++;
}


template<size_t N, typename T>
void expr_core_ptr<N, T>::release() {

    if(m_expr) {
        if(--m_expr->m_count == 0) {
            delete m_expr->m_core;
            delete m_expr;
        }
        m_expr = 0;
    }
}


template<size_t N, typename T>
inline expr_core_ptr<N, T>::counter::counter(expr_core_i<N, T> *core) :
    m_core(core), m_count(1) {
}


} // namespace iface
} // namespace libtensor

#endif // LIBTENSOR_IFACE_EXPR_CORE_I_H
