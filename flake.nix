# Development environment for ddx on Nix / NixOS.
#
#   nix develop        # or, with direnv: `direnv allow` once, then just `cd`
#
# Provides everything CONTRIBUTING.md asks for: a Rust toolchain (cargo, clippy,
# rustfmt), uv, and a Python for uv to use. It is a dev shell only — CI does not
# use Nix, and the published crates/wheel are built the usual way.
{
  description = "ddx — JAX-style automatic differentiation in SQL";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ];
      forAll = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAll (pkgs: {
        default = pkgs.mkShell {
          packages = with pkgs; [
            # Rust. nixpkgs' toolchain is newer than the 1.88 MSRV in Cargo.toml.
            cargo
            rustc
            clippy
            rustfmt
            rust-analyzer
            # Python: uv drives the venvs (python/ddxdb and tests/); maturin is
            # for `maturin develop`. python312 is what CI and the docs use.
            uv
            maturin
            python312
            # pyo3 and the crates' build scripts need a C compiler and pkg-config.
            pkg-config
            # `substrait` (ddx-ad) compiles its protos at build time.
            protobuf
          ];

          # Lets rust-analyzer find the standard library sources.
          RUST_SRC_PATH = "${pkgs.rustPlatform.rustLibSrc}";

          # uv's downloaded "managed" Pythons are generic-linux binaries that do
          # not run on NixOS. Only ever use the Nix-provided interpreter on PATH.
          UV_PYTHON_DOWNLOADS = "never";
          UV_PYTHON_PREFERENCE = "only-system";

          # PyPI wheels (jax, numpy, duckdb, datafusion, ...) are manylinux
          # binaries that expect libstdc++ on the loader path, which NixOS does
          # not provide globally. Appended, so an existing value is preserved.
          shellHook = pkgs.lib.optionalString pkgs.stdenv.hostPlatform.isLinux ''
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ]}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          '';
        };
      });

      formatter = forAll (pkgs: pkgs.nixpkgs-fmt);
    };
}
