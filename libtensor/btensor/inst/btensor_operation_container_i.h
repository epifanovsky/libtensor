#ifndef LIBTENSOR_BTENSOR_OPERATION_CONTAINER_I_H
#define LIBTENSOR_BTENSOR_OPERATION_CONTAINER_I_H

#include <vector>
#include <libtensor/btod/additive_btod.h>

namespace libtensor {


/** \brief Container for a block tensor operation and all associated objects

    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
class btensor_operation_container_i;

template<size_t N, typename T> class btensor_i;


/** \brief Container for a block tensor operation and all associated objects
        (specialized for double)

    \ingroup libtensor_expr
 **/
template<size_t N>
class btensor_operation_container_i<N, double> {
public:
    /** \brief Virtual constructor
     **/
    virtual ~btensor_operation_container_i() { }

    /** \brief Performs the block tensor operation into the given tensor
     **/
    virtual void perform(bool add, btensor_i<N, double> &bt) = 0;

    /** \brief Performs the block tensor operation into a new tensor
     **/
    virtual std::auto_ptr< btensor_i<N, double> > perform() = 0;

};


template<size_t N, typename T>
class btensor_operation_container_list {
private:
    std::vector< btensor_operation_container_i<N, T>* > m_list;

public:
    btensor_operation_container_list() {

    }

    ~btensor_operation_container_list() {
        for(size_t i = 0; i < m_list.size(); i++) delete m_list[i];
    }

    size_t size() const {
        return m_list.size();
    }

    void push_back(btensor_operation_container_i<N, T> *boc) {
        m_list.push_back(boc);
    }

    btensor_operation_container_i<N, T> &operator[](size_t i) {
        return *m_list[i];
    }

private:
    btensor_operation_container_list(
        const btensor_operation_container_list<N, T>&);

};


} // namespace libtensor

#endif // LIBTENSOR_BTENSOR_OPERATION_CONTAINER_I_H
