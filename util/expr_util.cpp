/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include "expr_util.h"
#include "fixedbv.h"
#include "bitvector.h"

exprt gen_zero(const typet &type)
{
  exprt result;

  const std::string type_id=type.id_string();

  result=exprt("constant", type);

  if(type_id=="rational" ||
     type_id=="real" ||
     type_id=="integer" ||
     type_id=="natural" ||
     type_id=="complex")
  {
    result.value("0");
  }
  else if(type_id=="unsignedbv" ||
          type_id=="signedbv" ||
          type_id=="floatbv" ||
          type_id=="fixedbv" ||
          type_id=="c_enum")
  {
    std::string value;
    unsigned width=bv_width(type);

    for(unsigned i=0; i<width; i++)
      value+='0';

    result.value(value);
  }
  else if(type_id=="bool")
  {
    result.make_false();
  }
  else if(type_id=="pointer")
  {
    result.value("NULL");
  }
  else
    result.make_nil();

  return result;
}

exprt gen_one(const typet &type)
{
  const std::string &type_id=type.id_string();
  exprt result=exprt("constant", type);

  if(type_id=="bool" ||
     type_id=="rational" ||
     type_id=="real" ||
     type_id=="integer" ||
     type_id=="natural" ||
     type_id=="complex")
  {
    result.value("1");
  }
  else if(type_id=="unsignedbv" ||
          type_id=="signedbv")
  {
    std::string value;
    for(int i=0; i<atoi(type.width().c_str())-1; i++)
      value+='0';
    value+='1';
    result.value(value);
  }
  else if(type_id=="fixedbv")
  {
    fixedbvt fixedbv;
    fixedbv.spec=to_fixedbv_type(type);
    fixedbv.from_integer(1);
    result=fixedbv.to_expr();
  }
  else if(type_id=="floatbv")
  {
    std::cerr << "floatbv unsupported, sorry" << std::endl;
    abort();
  }
  else
    result.make_nil();

  return result;
}

exprt gen_not(const exprt &op)
{
  return gen_unary("not", typet("bool"), op);
}

exprt gen_unary(const std::string &id, const typet &type, const exprt &op)
{
  exprt result(id, type);
  result.copy_to_operands(op);
  return result;
}

exprt gen_binary(const std::string &id, const typet &type, const exprt &op1, const exprt &op2)
{
  exprt result(id, type);
  result.copy_to_operands(op1, op2);
  return result;
}

exprt gen_binary(irep_idt &id, const typet &type, const exprt &op1, const exprt &op2)
{
  exprt result(id, type);
  result.copy_to_operands(op1, op2);
  return result;
}

exprt gen_and(const exprt &op1, const exprt &op2)
{
  return gen_binary("and", typet("bool"), op1, op2);
}

exprt gen_and(const exprt &op1, const exprt &op2, const exprt &op3)
{
  exprt result("and", typet("bool"));
  result.copy_to_operands(op1, op2, op3);
  return result;
}

exprt gen_or(const exprt &op1, const exprt &op2)
{
  return gen_binary("or", typet("bool"), op1, op2);
}

exprt gen_or(const exprt &op1, const exprt &op2, const exprt &op3)
{
  exprt result("or", typet("bool"));
  result.copy_to_operands(op1, op2, op3);
  return result;
}

exprt gen_implies(const exprt &op1, const exprt &op2)
{
  return gen_binary("=>", typet("bool"), op1, op2);
}

void gen_binary(exprt &expr, const std::string &id, bool default_value)
{
  if(expr.operands().size()==0)
  {
    if(default_value)
      expr.make_true();
    else
      expr.make_false();
  }
  else if(expr.operands().size()==1)
  {
    exprt tmp;
    tmp.swap(expr.op0());
    expr.swap(tmp);
  }
  else
  {
    expr.id(id);
    expr.type()=typet("bool");
  }
}

void gen_and(exprt &expr)
{
  gen_binary(expr, "and", true);
}

void gen_or(exprt &expr)
{
  gen_binary(expr, "or", false);
}

exprt symbol_expr(const symbolt &symbol)
{
  exprt tmp("symbol", symbol.type);
  tmp.identifier(symbol.name);
  return tmp;
}

pointer_typet gen_pointer_type(const typet &subtype)
{
  pointer_typet tmp;
  tmp.subtype()=subtype;
  return tmp;
}

exprt gen_address_of(const exprt &op)
{
  exprt tmp("address_of", gen_pointer_type(op.type()));
  tmp.copy_to_operands(op);
  return tmp;
}

void make_next_state(exprt &expr)
{
  Forall_operands(it, expr)
    make_next_state(*it);
    
  if(expr.id()=="symbol")
    expr.id("next_symbol");
}

exprt make_binary(const exprt &expr)
{
  const exprt::operandst &operands=expr.operands();

  if(operands.size()<=2) return expr;

  exprt previous=operands[0];
  
  for(unsigned i=1; i<operands.size(); i++)
  {
    exprt tmp=expr;
    tmp.operands().clear();
    tmp.operands().resize(2);
    tmp.op0().swap(previous);
    tmp.op1()=operands[i];
    previous.swap(tmp);
  }
  
  return previous;
}

