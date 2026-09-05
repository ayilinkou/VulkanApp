# Header-only for now: the cache is a template and the registry that will own it
# still names types that live in src/. It becomes a static library at the step
# that moves those.
engine_module(Assets)
