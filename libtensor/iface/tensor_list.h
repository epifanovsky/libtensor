#ifndef LIBTENSOR_IFACE_TENSOR_LIST_H
#define LIBTENSOR_IFACE_TENSOR_LIST_H

#include <map>
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
        virtual tensor_holder_base *clone() const = 0;
        virtual size_t get_n() const = 0;
        virtual const std::type_info &get_t() const = 0;
        virtual size_t get_tuid() const = 0;
    };

    template<size_t N, typename T>
    class tensor_holder : public tensor_holder_base {
    private:
        any_tensor<N, T> &m_t;
    public:
        tensor_holder(any_tensor<N, T> &t) : m_t(t) { }
        virtual ~tensor_holder() { }
        virtual tensor_holder_base *clone() const {
            return new tensor_holder<N, T>(*this);
        }
        virtual size_t get_n() const { return N; }
        virtual const std::type_info &get_t() const { return typeid(T); }
        virtual size_t get_tuid() const { return size_t(&m_t); }
        any_tensor<N, T> &get_tensor() const { return m_t; }
        bool tensor_equals(any_tensor<N, T> &other) { return &m_t == &other; }
    };

    typedef std::map<size_t, tensor_holder_base*> map_t;

private:
    map_t m_lst; //!< List of tensors

public:
    /** \brief Default constructor
     **/
    tensor_list();

    /** \brief Copy constructor
        \param tl Another tensor list.
     **/
    tensor_list(const tensor_list &tl);

    /** \brief Data transferring constructor
        \param tl Another tensor list, which will lose its data.
     **/
    tensor_list(tensor_list &tl, int);

    /** \brief Destructor
     **/
    ~tensor_list();

    /** \brief Assigns an ID to a tensor that is not on the list or returns
            the ID of an existing tensor
     **/
    template<size_t N, typename T>
    size_t get_tensor_id(any_tensor<N, T> &t);

    /** \brief Merge other tensor list into this one.
     **/
    void merge(const tensor_list &tl);

    /** \brief Returns the order of a tensor by previously assigned ID
     **/
    size_t get_tensor_order(size_t tid) const;

    /** \brief Returns the element type of a tensor by previously assigned ID
     **/
    const std::type_info &get_tensor_type(size_t tid) const;

    /** \brief Returns tensor by previously assigned ID
     **/
    template<size_t N, typename T>
    any_tensor<N, T> &get_tensor(size_t tid) const;

private:
    template<size_t N, typename T>
    bool check_type(map_t::const_iterator i) const;

};


template<size_t N, typename T>
size_t tensor_list::get_tensor_id(any_tensor<N, T> &t) {

    //  Find the tensor among those on the list

    tensor_holder<N, T> h(t);
    size_t tuid = h.get_tuid();
    if(m_lst.count(tuid) == 0) {
        m_lst[tuid] = h.clone();
    }
    return tuid;
}


template<size_t N, typename T>
any_tensor<N, T> &tensor_list::get_tensor(size_t tid) const {

    map_t::const_iterator i = m_lst.find(tid);
    if(i == m_lst.end()) {
        throw "Invalid tensor ID";
    }

    if(!check_type<N, T>(i)) {
        throw "Invalid tensor type";
    }

    return static_cast< tensor_holder<N, T>* >(i->second)->get_tensor();
}


template<size_t N, typename T>
bool tensor_list::check_type(map_t::const_iterator i) const {

    return (N == i->second->get_n() && typeid(T) == i->second->get_t());
}


} // namespace iface
} // namespace libtensor

#endif // LIBTENSOR_IFACE_TENSOR_LIST_H
