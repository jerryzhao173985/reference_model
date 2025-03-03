# Copyright (c) 2024-2025, ARM Limited.
# SPDX-License-Identifier: Apache-2.0


class TosaProfiles:
    TosaProINT = "tosa-pro-int"
    TosaProFP = "tosa-pro-fp"
    TosaExtBF16 = "tosa-ext-bf16"
    TosaExtControlFlow = "tosa-ext-controlflow"
    TosaExtDoubleRound = "tosa-ext-doubleround"
    TosaExtDynamic = "tosa-ext-dynamic"
    TosaExtFFT = "tosa-ext-fft"
    TosaExtFP8E4M3 = "tosa-ext-fp8e4m3"
    TosaExtFP8E5M2 = "tosa-ext-fp8e5m2"
    TosaExtInexactRound = "tosa-ext-inexactround"
    TosaExtInt16 = "tosa-ext-int16"
    TosaExtInt4 = "tosa-ext-int4"
    TosaExtVariable = "tosa-ext-variable"

    EXTENSION_COMPATIBLE_PROFILES = {
        TosaExtBF16: (TosaProFP,),
        TosaExtControlFlow: (TosaProINT, TosaProFP),
        TosaExtDoubleRound: (TosaProINT,),
        TosaExtDynamic: (TosaProINT, TosaProFP),
        TosaExtFFT: (TosaProFP,),
        TosaExtFP8E4M3: (TosaProFP,),
        TosaExtFP8E5M2: (TosaProFP,),
        TosaExtInexactRound: (TosaProINT,),
        TosaExtInt16: (TosaProINT,),
        TosaExtInt4: (TosaProINT,),
        TosaExtVariable: (TosaProINT, TosaProFP),
    }

    @staticmethod
    def profiles():
        return [TosaProfiles.TosaProINT, TosaProfiles.TosaProFP]

    @staticmethod
    def extensions():
        return [
            TosaProfiles.TosaExtBF16,
            TosaProfiles.TosaExtControlFlow,
            TosaProfiles.TosaExtDoubleRound,
            TosaProfiles.TosaExtDynamic,
            TosaProfiles.TosaExtFFT,
            TosaProfiles.TosaExtFP8E4M3,
            TosaProfiles.TosaExtFP8E5M2,
            TosaProfiles.TosaExtInexactRound,
            TosaProfiles.TosaExtInt16,
            TosaProfiles.TosaExtInt4,
            TosaProfiles.TosaExtVariable,
        ]

    @staticmethod
    def all():
        return TosaProfiles.profiles() + TosaProfiles.extensions()

    @staticmethod
    def isSupported(
        profiles_chosen, extensions_chosen, profiles_supported, extensions_required
    ):
        """
        Work out if the chosen profiles/extensions are supported.

        ANY matching chosen profiles in supported; AND
        ALL matching chosen extensions must be in required

        profiles_supported and extensions_required are usually read from the desc.json
        "test_requirements" entry
        """

        # Match any profile, but all extensions listed
        return (
            not profiles_supported
            or any([p in profiles_chosen for p in profiles_supported])
        ) and (
            not extensions_required
            or all([e in extensions_chosen for e in extensions_required])
        )

    @staticmethod
    def getCompatibleExtensions(profiles_chosen, extensions_chosen):
        """
        Returns the extensions from those chosen that are compatible
        with the chosen profiles
        """
        compatible_extensions = set()

        for extension in extensions_chosen:
            assert (
                extension in TosaProfiles.EXTENSION_COMPATIBLE_PROFILES
            ), f"Unknown extension {extension}"
            if any(
                [
                    p in profiles_chosen
                    for p in TosaProfiles.EXTENSION_COMPATIBLE_PROFILES[extension]
                ]
            ):
                compatible_extensions.add(extension)

        return list(compatible_extensions)

    PROFILES_EXTENSIONS_ALL = "all"
    PROFILES_EXTENSIONS_NONE = "none"

    @staticmethod
    def addArgumentsToParser(parser, all_extensions_default=True):
        """Add --profile & --extension arguments to given argparse object"""

        profile_default = TosaProfiles.PROFILES_EXTENSIONS_ALL
        parser.add_argument(
            "--profile",
            dest="profile",
            choices=TosaProfiles.profiles() + [TosaProfiles.PROFILES_EXTENSIONS_ALL],
            default=[profile_default],
            type=str,
            nargs="*",
            help=f"TOSA profile(s) - used for filtering tests (default is {profile_default})",
        )

        if all_extensions_default:
            extension_default = TosaProfiles.PROFILES_EXTENSIONS_ALL
            extension_help = (
                f" Use {TosaProfiles.PROFILES_EXTENSIONS_NONE} to choose no extensions."
            )
        else:
            extension_default = TosaProfiles.PROFILES_EXTENSIONS_NONE
            extension_help = (
                f" Use {TosaProfiles.PROFILES_EXTENSIONS_ALL} to choose all extensions."
            )

        parser.add_argument(
            "--extension",
            dest="extension",
            choices=TosaProfiles.extensions()
            + [
                TosaProfiles.PROFILES_EXTENSIONS_ALL,
                TosaProfiles.PROFILES_EXTENSIONS_NONE,
            ],
            default=[extension_default],
            type=str,
            nargs="*",
            help=f"TOSA extension(s) - used for filtering tests (default is {extension_default})."
            + extension_help,
        )

    def parseArguments(args, logger=None):
        """Validates and updates the profile and extension args"""

        if TosaProfiles.PROFILES_EXTENSIONS_ALL in args.profile:
            args.profile = TosaProfiles.profiles()
        else:
            # Remove duplicates from the list
            args.profile = list(set(args.profile))

        if TosaProfiles.PROFILES_EXTENSIONS_ALL in args.extension:
            args.extension = TosaProfiles.getCompatibleExtensions(
                args.profile, TosaProfiles.extensions()
            )
        elif TosaProfiles.PROFILES_EXTENSIONS_NONE in args.extension:
            args.extension = []
        else:
            # Remove duplicates from the list
            args.extension = list(set(args.extension))
            compatible_extensions = TosaProfiles.getCompatibleExtensions(
                args.profile, args.extension
            )
            if sorted(compatible_extensions) != sorted(args.extension):
                profiles_str = ", ".join(args.profile)
                for e in args.extension:
                    if e not in compatible_extensions and logger:
                        logger.warning(
                            f"Extension {e} is not compatible with the profile(s) chosen {profiles_str}"
                        )

                args.extension = compatible_extensions
