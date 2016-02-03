#ifndef LIBTENSOR_EXPR_CTF_BTENSOR_PLACEHOLDER_H
#define LIBTENSOR_EXPR_CTF_BTENSOR_PLACEHOLDER_H

#include <libtensor/expr/ctf_btensor/ctf_btensor.h>

namespace libtensor {
namespace expr {


class ctf_btensor_placeholder_base : public noncopyable {
public:
    virtual ~ctf_btensor_placeholder_base() { };

};


template<size_t N, typename T>
class ctf_btensor_placeholder :
    public ctf_btensor_placeholder_base, public any_tensor<N, T> {

public:
    static const char k_tensor_type[];

private:
    ctf_btensor<N, T> *m_bt; //!< Pointer to the real tensor

public:
    ctf_btensor_placeholder() : any_tensor<N, T>(*this), m_bt(0) {
    }

    virtual ~ctf_btensor_placeholder() {
        destroy_btensor();
    }

    virtual const char *get_tensor_type() const {
        return k_tensor_type;
    }

    void create_btensor(const block_index_space<N> &bis) {
        destroy_btensor();
        m_bt = new ctf_btensor<N, T>(bis);
    }

    void destroy_btensor() {
        delete m_bt;
        m_bt = 0;
    }

    bool is_empty() const {
        return m_bt == 0;
    }

    ctf_btensor<N, T> &get_btensor() const {
        if(m_bt == 0) throw 55;
        return *m_bt;
    }

public:
    /** \brief Converts any_tensor to btensor
     **/
    static ctf_btensor_placeholder<N, T> &from_any_tensor(any_tensor<N, T> &t) {
        return t.template get_tensor< ctf_btensor_placeholder<N, T> >();
    }

};


template<size_t N, typename T>
const char ctf_btensor_placeholder<N, T>::k_tensor_type[] =
    "ctf_btensor_placeholder";


} // namespace expr
} // namespace libtensor


#endif // LIBTENSOR_EXPR_CTF_BTENSOR_PLACEHOLDER_H
