#ifndef LIBTENSOR_EXPR_BTENSOR_PLACEHOLDER_H
#define LIBTENSOR_EXPR_BTENSOR_PLACEHOLDER_H

#include <libtensor/expr/btensor/btensor.h>

namespace libtensor {
namespace expr {


class btensor_placeholder_base : public noncopyable {
public:
    virtual ~btensor_placeholder_base() { };

};


template<size_t N, typename T>
class btensor_placeholder :
    public btensor_placeholder_base, public any_tensor<N, T> {

public:
    static const char k_tensor_type[];

private:
    btensor<N, T> *m_bt; //!< Pointer to the real tensor

public:
    btensor_placeholder() : any_tensor<N, T>(*this), m_bt(0) {
    }

    virtual ~btensor_placeholder() {
        destroy_btensor();
    }

    virtual const char *get_tensor_type() const {
        return k_tensor_type;
    }

    void create_btensor(const block_index_space<N> &bis) {
        destroy_btensor();
        m_bt = new btensor<N, T>(bis);
    }

    void destroy_btensor() {
        delete m_bt;
        m_bt = 0;
    }

    bool is_empty() const {
        return m_bt == 0;
    }

    btensor<N, T> &get_btensor() const {
        if(m_bt == 0) throw 55;
        return *m_bt;
    }

public:
    /** \brief Converts any_tensor to btensor
     **/
    static btensor_placeholder<N, T> &from_any_tensor(any_tensor<N, T> &t) {
        return t.template get_tensor< btensor_placeholder<N, T> >();
    }

};


template<size_t N, typename T>
const char btensor_placeholder<N, T>::k_tensor_type[] = "btensor_placeholder";


} // namespace expr
} // namespace libtensor


#endif // LIBTENSOR_EXPR_BTENSOR_PLACEHOLDER_H
