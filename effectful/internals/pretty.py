from effectful.ops.types import Term

try:
    from prettyprinter import install_extras, pretty_call, register_pretty

    install_extras({"dataclasses"})

    @register_pretty(Term)
    def pretty_term(value: Term, ctx):
        default_op_name = str(value.op)

        fresh_by_name = ctx.get("fresh_by_name") or {}
        new_ctx = ctx.assoc("fresh_by_name", fresh_by_name)

        fresh = fresh_by_name.get(default_op_name, {})
        fresh_by_name[default_op_name] = fresh

        fresh_ctr = fresh.get(value.op, len(fresh))
        fresh[value.op] = fresh_ctr

        op_name = str(value.op) + (f"!{fresh_ctr}" if fresh_ctr > 0 else "")
        return pretty_call(new_ctx, op_name, *value.args, **value.kwargs)

except ImportError:
    pass
