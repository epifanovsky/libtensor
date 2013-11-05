#include <string>
#include <libtensor/core/scalar_transf_double.h>
#include "node_assign.h"
#include "node_ident.h"
#include "node_transform.h"
#include "unary_node_base.h"
#include "nary_node_base.h"
#include "print_node.h"

namespace libtensor {
namespace expr {


void print_node(const node &n, std::ostream &os, size_t indent) {

    std::string ind(indent, ' ');
    const node_assign *na = dynamic_cast<const node_assign*>(&n);
    const node_ident *ni = dynamic_cast<const node_ident*>(&n);
    const unary_node_base *n1 = dynamic_cast<const unary_node_base*>(&n);
    const nary_node_base *nn = dynamic_cast<const nary_node_base*>(&n);
    if(ni) {
        os << ind << "( ident " << (void*)ni->get_tid() << " )" << std::endl;
    } else {
        os << ind << "( " << n.get_op();
        if(na) {
            os << " " << (void*)na->get_tid() << (na->is_add() ? " (+)" : "")
                << std::endl;
            print_node(na->get_rhs(), os, indent + 2);
        } else if(n1) {
            if(n1->get_op().compare("transform") == 0) {
                const node_transform_base *ntr0 =
                    dynamic_cast<const node_transform_base*>(n1);
                if(ntr0) {
                    const std::vector<size_t> &p = ntr0->get_perm();
                    os << "  [";
                    for(size_t i = 0; i + 1 < p.size(); i++) os << p[i] << ",";
                    if(p.size() > 0) os << p[p.size() - 1];
                    else os << "*";
                    os << "]";
                    const node_transform<double> *ntrd =
                        dynamic_cast< const node_transform<double>* >(ntr0);
                    if(ntrd) {
                        os << " " << ntrd->get_coeff().get_coeff();
                    }
                }
            }
            os << std::endl;
            print_node(n1->get_arg(), os, indent + 2);
        } else if(nn) {
            os << std::endl;
            for(size_t i = 0; i < nn->get_nargs(); i++) {
                print_node(nn->get_arg(i), os, indent + 2);
            }
        } else {
            os << " ???" << std::endl;
        }
        os << ind << ")" << std::endl;
    }
}


} // namespace expr
} // namespace libtensor
