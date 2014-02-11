#include <string>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/expr/dag/node_ident.h>
#include <libtensor/expr/dag/node_transform.h>
#include "print_node.h"

namespace libtensor {
namespace expr {


void print_node(const node &n, std::ostream &os) {

    if(n.get_op().compare(node_ident::k_op_type) == 0) {
        const node_ident &n1 = dynamic_cast<const node_ident&>(n);
        os << "(ident <" << n1.get_n() << ", " << n1.get_type().name() << ">)";
    } else {
        os << "(" << n.get_op();
        if(n.get_op().compare(node_transform_base::k_op_type) == 0) {
            const node_transform_base &n1 =
                    dynamic_cast<const node_transform_base&>(n);
            const std::vector<size_t> &p = n1.get_perm();
            os << "  [";
            for(size_t i = 0; i + 1 < p.size(); i++) os << p[i] << ", ";
            if(p.size() > 0) os << p[p.size() - 1];
            else os << "*";
            os << "]";
            if(n1.get_type() == typeid(double)) {
                const node_transform<double> &n2 =
                    dynamic_cast< const node_transform<double>& >(n1);
                os << " " << n2.get_coeff().get_coeff();
            }
        }
        os << ")";
    }
}


} // namespace expr
} // namespace libtensor
