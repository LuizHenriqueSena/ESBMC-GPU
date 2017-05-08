/*******************************************************************\

Module: Symbolic Execution

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <irep2.h>

#include <assert.h>

#include <expr_util.h>

#include "goto_symex.h"

void goto_symext::symex_other(void)
{
  const goto_programt::instructiont &instruction=*cur_state->source.pc;

  expr2tc code2 = instruction.code;

  if (is_code_expression2t(code2))
  {
    // This is some kind of expression, that gets evaluated. Given the GOTO
    // transformation, it can't contain any function calls, or have any other
    // side effects, so it's essentially a no-op.
    // However, it /might/ contain a dereference. And that dereference might
    // trigger an assertion failure due to an invalid access. And that kind of
    // tests is /definitely/ in the regression test suite (01_cmbc_String2).
    // So, dereference the expression here to collect any assertions it may
    // cause.
    const code_expression2t &expr = to_code_expression2t(code2);
    expr2tc operand = expr.operand;
    dereference(operand, false, false);
  }
  else if (is_code_cpp_del_array2t(code2) || is_code_cpp_delete2t(code2))
  {
    expr2tc deref_code(code2);

    replace_dynamic_allocation(deref_code);
    replace_nondet(deref_code);
    dereference(deref_code, false);

    symex_cpp_delete(deref_code);
  }
  else if (is_code_free2t(code2))
  {
    symex_free(code2);
  }
  else if (is_code_printf2t(code2))
  {
    replace_dynamic_allocation(code2);
    replace_nondet(code2);
    dereference(code2, false);
    symex_printf(expr2tc(), code2);
  }
  else if (is_code_decl2t(code2))
  {
    replace_dynamic_allocation(code2);
    replace_nondet(code2);
    dereference(code2, false);

    const code_decl2t &decl_code = to_code_decl2t(code2);

    // just do the L2 renaming to preseve locality
    const irep_idt &identifier = decl_code.value;

    // Generate dummy symbol as a vehicle for renaming.
    symbol2tc l1_sym(get_empty_type(), identifier);

    cur_state->top().level1.get_ident_name(l1_sym);
    symbol2t &l1_symbol = to_symbol2t(l1_sym);

    // increase the frame if we have seen this declaration before
    while(cur_state->top().declaration_history.find(renaming::level2t::name_record(l1_symbol))!=
          cur_state->top().declaration_history.end())
    {
      unsigned &index = cur_state->variable_instance_nums[identifier];
      cur_state->top().level1.rename(l1_sym, ++index);
      l1_symbol.level1_num = index;
    }

    renaming::level2t::name_record tmp_name(l1_symbol);
    cur_state->top().declaration_history.insert(tmp_name);
    cur_state->top().local_variables.insert(tmp_name);

    // seen it before?
    // it should get a fresh value
    if (cur_state->level2.current_number(l1_sym) != 0)
    {
      // Dummy assignment - blank constant value isn't considered for const
      // propagation, variable number will be bumped to result in a new free
      // variable. Invalidates l1_symbol reference?
      cur_state->level2.make_assignment(l1_sym, expr2tc(), expr2tc());
    }
  }
  else if (is_code_asm2t(code2))
  {
    // Assembly statement -> do nothing.
    return;
  }
  else
    throw "goto_symext: unexpected statement: " + get_expr_id(code2);
}

