#ifndef SPARSITY_EXPR_H
#define SPARSITY_EXPR_H

#include "subspace.h"
#include <deque>

namespace libtensor {

//Forward declaration
template<size_t N>
class sparse_bispace; 

//Forward declaration for partial specialization
//M = order of the bispace on the LHS of the first operator%(...) invocation
//N = number of bispaces coupled by sparsity to the right-most member of the parent bispace  
template<size_t M,size_t N>
class sparsity_expr;

//Base case
template<size_t M>
class sparsity_expr<M,1> {
private:
    //The first bispace that called operator%(...) to instantiate this object
    const sparse_bispace<M>& m_parent_bispace;
    const sparse_bispace<1>& m_cur_bispace;
    
    //Constructor - private because we only want instances of this class created by sparse_bispace<M>::operator%(...)
    sparsity_expr(const sparse_bispace<M>& parent_bispace, const sparse_bispace<1>& rhs) : m_parent_bispace(parent_bispace),m_cur_bispace(rhs) {}

    //Copy constructor is private, so can ONLY be actually used in code 
    //by calling operator<<(...) to extract a sparse bispace object
    sparsity_expr(const sparsity_expr<M,1>& rhs) : m_parent_bispace(rhs.m_parent_bispace), m_cur_bispace(rhs.m_cur_bispace) {}

    //Internal method for recursively constructing a list of all subspaces involved in the expression
    void retrieve_subspaces(std::deque<subspace>& subspaces) const;
public:
    //Resolve this expression into a true sparse_bispace
    //Implemented in sparse_bispace.h
    sparse_bispace<2> operator<<(const std::vector< sequence<2,size_t> >& sig_blocks);

    //Chain this expression with another bispace to create a higher order sparsity expr
    //sparsity_expr<M,N+1> operator%(const sparse_bispace<1>& rhs);
    sparsity_expr<M,2> operator%(const sparse_bispace<1>& rhs);

    //Friend sparse_bispace so that it can create instances of this class
    template<size_t P> 
    friend class sparse_bispace;

    //Friend higher orders for recursion
    template<size_t P,size_t Q>
    friend class sparsity_expr;
};

template<size_t M>
sparsity_expr<M,2> sparsity_expr<M,1>::operator%(const sparse_bispace<1>& rhs)
{
    return sparsity_expr<M,2>(*this,rhs);
}


//General case
template<size_t M,size_t N>
class sparsity_expr {
private:
    const sparsity_expr<M,N-1>& m_sub_expr;
    const sparse_bispace<1>& m_cur_bispace;
    const sparse_bispace<M>& m_parent_bispace;

    //Constructor - private because we only want instances of this class created by sparse_bispace<M>::operator%(...)
    //Used 
    sparsity_expr(const sparsity_expr<M,N-1>& sub_expr,const sparse_bispace<1>& cur_subspace) : m_sub_expr(sub_expr), m_cur_bispace(cur_subspace),m_parent_bispace(sub_expr.m_parent_bispace) {}
    
    //Copy constructor is private, so can ONLY be actually used in code 
    //by calling operator<<(...) to extract a sparse bispace object
    sparsity_expr(const sparsity_expr<M,N>& rhs) : m_sub_expr(rhs.m_sub_expr), m_cur_bispace(rhs.m_cur_bispace), m_parent_bispace(rhs.m_parent_bispace) {}

    //Internal method for recursively constructing a list of all subspaces involved in the expression
    void retrieve_subspaces(std::deque<subspace>& subspaces) const; 
public:
    //Resolve this expression into a true sparse_bispace
    //Implemented in sparse_bispace.h
    sparse_bispace<M+N> operator<<(const std::vector< sequence<N+1,size_t> >& sig_blocks);

    //Chain this expression with another bispace to create a higher order sparsity expr
    //sparsity_expr<M,N+1> operator%(const sparse_bispace<1>& rhs);
    sparsity_expr<M,N+1> operator%(const sparse_bispace<1>& rhs) const;

    //Friend other orders to allow recursion
    template<size_t P,size_t Q>
    friend class sparsity_expr;
};

template<size_t M,size_t N>
sparsity_expr<M,N+1> sparsity_expr<M,N>::operator%(const sparse_bispace<1>& rhs) const
{
    return sparsity_expr<M,N+1>(*this,rhs);
}

} // namespace libtensor


#endif /* SPARSITY_EXPR_H */
