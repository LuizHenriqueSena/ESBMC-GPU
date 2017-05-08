/*******************************************************************\

Module: Symbolic Execution

Author: Daniel Kroening, kroening@kroening.com
		Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#include <irep2.h>
#include <migrate.h>

#include <assert.h>
#include <global.h>
#include <map>
#include <sstream>

#include <i2string.h>
#include "../util/expr_util.h"

#include "reachability_tree.h"
#include "execution_state.h"
#include "goto_symex_state.h"
#include "goto_symex.h"

goto_symex_statet::goto_symex_statet(renaming::level2t &l2, value_sett &vs,
                                     const namespacet &_ns)
    : guard(), level2(l2), value_set(vs), ns(_ns)
{
  use_value_set = true;
  depth = 0;
  thread_ended = false;
  guard.make_true();
}

goto_symex_statet::goto_symex_statet(const goto_symex_statet &state,
                                     renaming::level2t &l2,
                                     value_sett &vs)
  : level2(l2), value_set(vs), ns(state.ns)
{
  *this = state;
}

goto_symex_statet &
goto_symex_statet::operator=(const goto_symex_statet &state)
{
  depth = state.depth;
  thread_ended = state.thread_ended;
  guard = state.guard;
  global_guard = state.global_guard;
  source = state.source;
  variable_instance_nums = state.variable_instance_nums;
  unwind_map = state.unwind_map;
  function_unwind = state.function_unwind;
  use_value_set = state.use_value_set;
  call_stack = state.call_stack;
  return *this;
}

void goto_symex_statet::initialize(const goto_programt::const_targett & start, const goto_programt::const_targett & end, const goto_programt *prog, unsigned int thread_id)
{
  new_frame(thread_id);

  source.is_set=true;
  source.thread_nr = thread_id;
  source.pc=start;
  source.prog = prog;
  top().end_of_function=end;
  top().calling_location=symex_targett::sourcet(top().end_of_function, prog);
}

bool goto_symex_statet::constant_propagation(const expr2tc &expr) const
{
  static unsigned int with_counter=0;

  // Don't permit const propagaion of infinite-size arrays. They're going to
  // be special modelling arrays that require special handling either at SMT
  // or some other level, so attempting to optimse them is a Bad Plan (TM).
  if (is_array_type(expr) && to_array_type(expr->type).size_is_infinite)
    return false;

  if (is_nil_expr(expr)) {
    return true; // It's fine to constant propagate something that's absent.
  } else if (is_constant_expr(expr)) {
    return true;
  }
  else if (is_symbol2t(expr) && to_symbol2t(expr).thename == "NULL")
  {
    // Null is also essentially a constant.
    return true;
  }
  else if (is_address_of2t(expr))
  {
    return constant_propagation_reference(to_address_of2t(expr).ptr_obj);
  }
  else if (is_typecast2t(expr))
  {
    return constant_propagation(to_typecast2t(expr).from);
  }
  else if (is_add2t(expr))
  {
    forall_operands2(it, idx, expr)
      if(!constant_propagation(*it))
        return false;

    return true;
  }
  else if (is_constant_array_of2t(expr))
  {
    const expr2tc &init = to_constant_array_of2t(expr).initializer;
    if (is_constant_expr(init) && !is_bool_type(init))
      return true;
  }
  else if (is_with2t(expr))
  {
    // Keeping additional with data achieves nothing; no code in ESBMC inspects
    // with chains to extract data from them.
    // FIXME: actually benchmark this and look at timing results, it may be
    // important benchmarks (i.e. TACAS) work better with some propagation
    return false;
    with_counter++;
  }
  else if (is_constant_struct2t(expr))
  {
    forall_operands2(it, idx, expr)
      if(!constant_propagation(*it))
        return false;

    return true;
  }
  else if (is_constant_union2t(expr))
  {
    const expr2tc *e = expr->get_sub_expr(0);
    if (e == NULL)
      return false;
    if (is_nil_expr(*e))
      return false;
    if (expr->get_sub_expr(1) != NULL) // Ensure only one operand (?????)
                                       // Preserves previous behaviour.
      return false;

    return constant_propagation(*e);
  }

  /* No difference
  else if(expr.id()==exprt::equality)
  {
    if(expr.operands().size()!=2)
	  throw "equality expects two operands";

    return (constant_propagation(expr.op0()) ||
           constant_propagation(expr.op1()));

  }
  */

  return false;
}

bool goto_symex_statet::constant_propagation_reference(const expr2tc &expr)const
{

  if (is_symbol2t(expr))
    return true;
  else if (is_index2t(expr))
  {
    const index2t &index = to_index2t(expr);
    return constant_propagation_reference(index.source_value) &&
           constant_propagation(index.index);
  }
  else if (is_member2t(expr))
  {
    return constant_propagation_reference(to_member2t(expr).source_value);
  }
  else if (is_constant_string2t(expr))
    return true;

  return false;
}

void goto_symex_statet::assignment(
  expr2tc &lhs,
  const expr2tc &rhs,
  bool record_value)
{
  assert(is_symbol2t(lhs));
  symbol2t &lhs_sym = to_symbol2t(lhs);

  // identifier should be l0 or l1, make sure it's l1

  assert(lhs_sym.rlevel != symbol2t::level2 &&
         lhs_sym.rlevel != symbol2t::level2_global);

  if (lhs_sym.rlevel == symbol2t::level0)
    top().level1.get_ident_name(lhs);

  expr2tc l1_lhs = lhs;

  expr2tc const_value;
  if(record_value && constant_propagation(rhs))
    const_value = rhs;
  else
    const_value = expr2tc();

  level2.make_assignment(lhs, const_value, rhs);

  if(use_value_set)
  {
    // update value sets
    expr2tc l1_rhs = rhs; // rhs is const; Rename into new container.
    level2.get_original_name(l1_rhs);

    value_set.assign(l1_lhs, l1_rhs);
  }
}

void goto_symex_statet::rename(expr2tc &expr)
{
  // rename all the symbols with their last known value

  if (is_nil_expr(expr))
    return;

  if (is_symbol2t(expr))
  {
    type2tc origtype = expr->type;
    top().level1.rename(expr);
    level2.rename(expr);
    fixup_renamed_type(expr, origtype);
  }
  else if (is_address_of2t(expr))
  {
    address_of2t &addrof = to_address_of2t(expr);
    rename_address(addrof.ptr_obj);
  }
  else
  {
    // do this recursively
    Forall_operands2(it, idx, expr) {
      rename(*it);
    }
  }
}

void goto_symex_statet::rename_address(expr2tc &expr)
{
  // rename all the symbols with their last known value

  if (is_nil_expr(expr))
  {
    return;
  }
  else if(is_symbol2t(expr))
  {
    // only do L1
    type2tc origtype = expr->type;
    top().level1.rename(expr);
    fixup_renamed_type(expr, origtype);

    // Realloc hacks: The l1 name may need to change slightly when we realloc
    // a pointer, so that l2 renaming still points at the same piece of data,
    // but so that the address compares differently to previous address-of's.
    // Do this by bumping the l2 number in the l1 name, if it's been realloc'd.
    if (realloc_map.find(expr) != realloc_map.end()) {
      symbol2t &sym = to_symbol2t(expr);
      sym.level2_num = realloc_map[expr];
    }
  }
  else if (is_index2t(expr))
  {
    index2t &index = to_index2t(expr);
    rename_address(index.source_value);
    rename(index.index);
  }
  else
  {
    // do this recursively
    Forall_operands2(it, idx, expr) {
      rename_address(*it);
    }
  }
}

void goto_symex_statet::fixup_renamed_type(expr2tc &expr,
                                           const type2tc &orig_type)
{
  if (is_code_type(orig_type)) {
    return;
  } else if (is_pointer_type(orig_type)) {
    assert(is_pointer_type(expr));

    // Grab pointer types
    const pointer_type2t &orig = to_pointer_type(orig_type);
    const pointer_type2t &newtype = to_pointer_type(expr->type);

    type2tc origsubtype = orig.subtype;
    type2tc newsubtype = newtype.subtype;

    // Handle symbol subtypes -- we can't rename these, because there are (some)
    // pointers to incomplete types, that here we end up trying to get a
    // concrete type for. Which is incorrect.
    // So instead, if one of the subtypes is a symbol type, and it isn't
    // identical to the other type, insert a typecast. This might lead to some
    // needless casts, but what the hell.
    if (is_symbol_type(origsubtype) || is_symbol_type(newsubtype)) {
      if (origsubtype != newsubtype) {
        expr = typecast2tc(orig_type, expr);
      }
      return;
    }

    // Cease caring about anything that points at code types: pointer arithmetic
    // applied to this is already broken.
    if (is_code_type(origsubtype) || is_code_type(newsubtype))
      return;

    if (origsubtype == newsubtype)
      return;

    // Fetch the (bit) size of the pointer subtype.
    unsigned int origsize, newsize;

    if (is_empty_type(origsubtype))
      origsize = 8;
    else
      origsize = origsubtype->get_width();

    if (is_empty_type(newsubtype))
      newsize = 8;
    else
      newsize = newsubtype->get_width();

    // If the renaming process has changed the size of the pointer subtype, this
    // will break all kinds of pointer arith; insert a cast.
    if (origsize != newsize) {
      expr = typecast2tc(orig_type, expr);
    }
  } else if (is_scalar_type(orig_type) && is_scalar_type(expr->type)) {
    // If we're a BV and have changed size, then we're quite likely to cause
    // an SMT problem later on. Immediately cast. Also if we've gratuitously
    // changed sign.
    if (orig_type->get_width() != expr->type->get_width() ||
                    (is_bv_type(orig_type) && is_bv_type(expr->type) &&
                    orig_type->type_id != expr->type->type_id)) {
      expr = typecast2tc(orig_type, expr);
    }
  }
}

void goto_symex_statet::get_original_name(expr2tc &expr) const
{

  if (is_nil_expr(expr))
    return;

  Forall_operands2(it, idx, expr)
    get_original_name(*it);

  if (is_symbol2t(expr))
  {
    level2.get_original_name(expr);
    top().level1.get_original_name(expr);
  }
}

void goto_symex_statet::print_stack_trace(unsigned int indent) const
{
  call_stackt::const_reverse_iterator it;
  symex_targett::sourcet src;
  std::string spaces = std::string("");
  unsigned int i;

  for (i = 0; i < indent; i++)
    spaces += " ";

  // Iterate through each call frame printing func name and location.
  src = source;
  for (it = call_stack.rbegin(); it != call_stack.rend(); it++) {
    if (it->function_identifier == "") { // Top level call
      std::cout << spaces << "init" << std::endl;
    } else {
      std::cout << spaces << it->function_identifier.as_string();
      std::cout << " at " << src.pc->location.get_file();
      std::cout << " line " << src.pc->location.get_line();
      std::cout << std::endl;
    }

    src = it->calling_location;
  }

  if (!thread_ended) {
    std::cout << spaces << "Next instruction to be executed:" << std::endl;
    source.prog->output_instruction(ns, "", std::cout, source.pc, true, false);
  }

  return;
}

std::vector<dstring>
goto_symex_statet::gen_stack_trace(void) const
{
  std::vector<dstring> trace;
  call_stackt::const_reverse_iterator it;
  symex_targett::sourcet src;

  // Format is a vector of strings, each recording a particular function
  // invocation.

  for (it = call_stack.rbegin(); it != call_stack.rend(); it++) {
    src = it->calling_location;

    if (it->function_identifier == "") { // Top level call
      break;
    } else if (it->function_identifier == "c::main" &&
               src.pc->location == get_nil_irep()) {
      trace.push_back("<main invocation>");
    } else {
      std::string loc = it->function_identifier.as_string();
      loc += " at " + src.pc->location.get_file().as_string();
      loc += " line " + src.pc->location.get_line().as_string();
      trace.push_back(loc);
    }
  }

  return trace;
}
