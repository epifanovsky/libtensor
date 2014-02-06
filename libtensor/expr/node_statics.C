#include "node_add.h"
#include "node_assign.h"
#include "node_contract.h"
#include "node_diag.h"
#include "node_dirsum.h"
#include "node_div.h"
#include "node_ident_base.h"
#include "node_null.h"
#include "node_scalar_base.h"
#include "node_symm_base.h"
#include "node_transform_base.h"

namespace libtensor {
namespace expr {


const char node_add::k_op_type[] = "add";
const char node_assign::k_op_type[] = "assign";
const char node_contract::k_op_type[] = "contract";
const char node_diag::k_op_type[] = "diag";
const char node_dirsum::k_op_type[] = "dirsum";
const char node_div::k_op_type[] = "div";
const char node_ident_base::k_op_type[] = "ident";
const char node_null::k_op_type[] = "null";
const char node_scalar_base::k_op_type[] = "scalar";
const char node_symm_base::k_op_type[] = "symm";
const char node_transform_base::k_op_type[] = "transform";


} // namespace expr
} // namespace libtensor
