{
    macro_rules! set_args (
        ($kernel:expr, $i: expr) => {()};
        
        ($kernel:expr, $i:expr, $arg:expr) => {{
            $kernel.set_arg($i, $arg);
        }};
        
        ($kernel:expr, $i:expr, $arg:expr $(, $args:expr)*) => {{
            $kernel.set_arg($i, $arg);
            set_args!($kernel, $i + 1 $(, $args)*);
        }};
    )
        
        macro_rules! execute (
            ($kernel:ident [$global:expr, $local:expr] ,
             $($args:expr),*) => {{
                $kernel.set_arg(0, &0);
                $kernel.set_arg(1, &0);
                set_args!($kernel, 2, $($args),*);
                $kernel.execute($global, $local)
            }}
        )
}
