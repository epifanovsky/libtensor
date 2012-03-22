#ifndef LIBTENSOR_LOOP_LIST_NODE_H
#define LIBTENSOR_LOOP_LIST_NODE_H

namespace libtensor {


/** \brief Structure to store increments in each array in a loop

    \ingroup libtensor_kernels
 **/
template<size_t N, size_t M>
struct loop_list_node {
private:
    size_t m_weight; //!< Number of iterations in the loop
    size_t m_stepa[N]; //!< Increments in the input arrays
    size_t m_stepb[M]; //!< Increments in the output arrays

public:
    /** \brief Default constructor
     **/
    loop_list_node() : m_weight(0) {
        init_incs();
    }

    /** \brief Initializing constructor
     **/
    loop_list_node(size_t weight) : m_weight(weight) {
        init_incs();
    }

    size_t &weight() { return m_weight; }
    size_t weight() const { return m_weight; }
    size_t &stepa(size_t i) { return m_stepa[i]; }
    size_t stepa(size_t i) const { return m_stepa[i]; }
    size_t &stepb(size_t i) { return m_stepb[i]; }
    size_t stepb(size_t i) const { return m_stepb[i]; }

private:
    void init_incs() {
        for(size_t i = 0; i < N; i++) m_stepa[i] = 0;
        for(size_t i = 0; i < M; i++) m_stepb[i] = 0;
    }

};


} // namespace libtensor

#endif // LIBTENSOR_LOOP_LIST_NODE_H
