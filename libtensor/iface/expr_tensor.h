#ifndef LIBTENSOR_EXPR_TENSOR_H
#define LIBTENSOR_EXPR_TENSOR_H

namespace libtensor {

template<size_t N, typename T>
class any_tensor;

template<size_t N, typename T>
class expr_core_i;


template<size_t N, typename T>
class ident_core : public expr_core_i<N, T> {

    ident_core(any_tensor<N, T> &t);
};

/** \brief Interface for tensor factories
 **/
template<size_t N, typename T>
class tensor_factory_i {
public:
    virtual ~tensor_factory_i() {}

    // Clone factory
    virtual tensor_factor_i<N, T> *clone() const;

    // Create empty tensor
    virtual any_tensor<N, T> *create() const;

    // Create tensor by evaluating an expression
    virtual any_tensor<N, T> *create(expr_core_i<N, T> &expr) const;

    // Check expr for consistency with the tensor to be created by the factory
    virtual bool check(expr_core_i<N, T> &core) const;
};


/** \brief User-friendly block %tensor

    After creation expr_tensor only contains tensor factory, expression core is
    empty.


    \ingroup libtensor_iface
 **/
template<size_t N, typename T>
class expr_tensor {
private:
    tensor_factory_i<N, T> *m_fac; //!< Factory to create tensor and evaluate expression
    expr_core_i<N, T> *m_expr; //!< Expression
    any_tensor<N, T> *m_tensor; //!< Tensor object

public:
    //! Constructor
    expr_tensor(const tensor_factory_i<N, T> &fac) :
        m_fac(fac->clone()), m_expr(0), m_tensor(0) {

    }

    expr_tensor<N, T> &operator=(const expr_core_i<N, T> &expr) {
        if (m_tensor != 0) { delete m_tensor; m_tensor = 0; }
        if (m_expr != 0) { delete m_expr; m_expr = 0; }
        m_expr = expr.clone();

        return *this;
    }

    any_tensor<N, T> &get_tensor() {
        init_tensor();
        return *m_tensor;
    }

    const expr_core_i<N, T> &get_expr() {
        init_expr();
        return *m_expr;
    }

    /** \brief Attaches a label to this %tensor and returns it as a
            labeled %tensor
     **/
    labeled_tensor<N, T> operator()(const letter_expr<N> &expr);

private:
    void init_expr() {
        if (m_expr != 0) return;

        if (m_tensor != 0) {
            m_expr = new ident_core<N, T>(m_tensor);
        }
        else {
            init_tensor();
        }
    }

    void init_tensor() {
        if (m_tensor != 0) return;

        if (m_expr != 0) {
            m_tensor = m_fac->create(*m_expr);
        }
        else {
            m_tensor = m_fac->create();
            init_expr();
        }
    }
};


} // namespace libtensor

#endif // LIBTENSOR_EXPR_TENSOR_H

