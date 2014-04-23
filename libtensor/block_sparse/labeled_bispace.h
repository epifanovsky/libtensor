#ifndef LABELED_BISPACE_H
#define LABELED_BISPACE_H

#include "sparse_bispace.h"
#include "../expr/iface/label.h"

namespace libtensor {

class labeled_bispace
{
private:
    class abstract_base
    {
    public:
        virtual abstract_base* clone() const = 0;
    };

    template<size_t N>
    class wrapper : public abstract_base
    {
    private:
        sparse_bispace<N> m_bispace; 
        expr::label<N> m_label;
    public:
        wrapper(const sparse_bispace<N>& bispace,const expr::label<N> label) : m_bispace(bispace),m_label(label) {}
        abstract_base* clone() const { return new wrapper<N>(m_bispace,m_label); }
    };
    abstract_base* m_content;
public:
    template<size_t N>
    explicit labeled_bispace(const sparse_bispace<N>& bispace,const expr::label<N> label) : m_content(new wrapper<N>(bispace,label))  {}

    labeled_bispace(const labeled_bispace& rhs) : m_content(rhs.m_content->clone()) {}
    labeled_bispace& operator=(const labeled_bispace& rhs) { delete m_content; m_content = rhs.m_content->clone(); return *this; }
    ~labeled_bispace() { delete m_content; }
};

} // namespace libtensor
#endif /* LABELED_BISPACE_H */
