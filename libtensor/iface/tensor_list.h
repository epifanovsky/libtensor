#ifndef LIBTENSOR_IFACE_TENSOR_LIST_H
#define LIBTENSOR_IFACE_TENSOR_LIST_H

#include <vector>
#include "any_tensor.h"

namespace libtensor {
namespace iface {


/** \brief Maintains a list of any_tensor objects

    \ingroup libtensor_iface
 **/
class tensor_list {
private:
    class tensor_holder_base {
    public:
        virtual ~tensor_holder_base() { }
        virtual size_t get_n() const = 0;
        virtual const std::type_info &get_t() const = 0;
    };

    template<size_t N, typename T>
    class tensor_holder : public tensor_holder_base {
    private:
        any_tensor<N, T> &m_t;
    public:
        tensor_holder(any_tensor<N, T> &t) : m_t(t) { }
        virtual ~tensor_holder() { }
        virtual size_t get_n() const { return N; }
        virtual const std::type_info &get_t() const { return typeid(T); }
        any_tensor<N, T> &get_tensor() { return m_t; }
        bool tensor_equals(any_tensor<N, T> &other) { return &m_t == &other; }
    };

private:
    std::vector<tensor_holder_base*> m_lst; //!< List of tensors

public:
    /** \brief Destructor
     **/
    ~tensor_list();

    /** \brief Assigns an ID to a tensor that is not on the list or returns
            the ID of an existing tensor
     **/
    template<size_t N, typename T>
    unsigned get_tensor_id(any_tensor<N, T> &t);

    /** \brief Returns the order of a tensor by previously assigned ID
     **/
    size_t get_tensor_order(unsigned tid) const;

    /** \brief Returns the element type of a tensor by previously assigned ID
     **/
    const std::type_info &get_tensor_type(unsigned tid) const;

    /** \brief Returns tensor by previously assigned ID
     **/
    template<size_t N, typename T>
    any_tensor<N, T> &get_tensor(unsigned tid);

private:
    template<size_t N, typename T>
    bool check_type(unsigned tid);

};


template<size_t N, typename T>
unsigned tensor_list::get_tensor_id(any_tensor<N, T> &t) {

    //  Find the tensor among those on the list

    for(size_t i = 0; i < m_lst.size(); i++) {
        if(check_type<N, T>(i)) {
            tensor_holder<N, T> *h =
                static_cast< tensor_holder<N, T>* >(m_lst[i]);
            if(h->tensor_equals(t)) return i;
        }
    }

    //  Or add a new record

    unsigned tid = m_lst.size();
    m_lst.push_back(new tensor_holder<N, T>(t));
    return tid;
}


template<size_t N, typename T>
any_tensor<N, T> &tensor_list::get_tensor(unsigned tid) {

    if(tid >= m_lst.size()) {
        throw 0;
    }

    if(!check_type<N, T>(tid)) {
        throw 0;
    }

    return static_cast< tensor_holder<N, T>* >(m_lst[tid])->get_tensor();
}


template<size_t N, typename T>
bool tensor_list::check_type(unsigned tid) {

    return (N == m_lst[tid]->get_n() && typeid(T) == m_lst[tid]->get_t());
}


} // namespace iface
} // namespace libtensor

#endif // LIBTENSOR_IFACE_TENSOR_LIST_H
