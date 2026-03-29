import os
import json
import pickle
import re

from .utils import *
from .oxford_pets import OxfordPets
from .dtd import DescribableTextures as DTD


class Country211(DatasetBase):
    """
    Country211 dataset loader for MTIL.
    Expected structure:
      <root>/country211/
        ├─ train/<ISO2>/*.jpg
        ├─ valid/<ISO2>/*.jpg
        └─ test/<ISO2>/*.jpg

    Where <ISO2> are ISO-3166 alpha-2 country codes (e.g., AD, US, CN).
    Class names are mapped from ISO2 codes to provided human-readable names.
    Optionally, a JSON mapping file can override defaults:
      <root>/country211/iso2_to_name.json  => {"AD": "Andorra", ...}
    """

    dataset_dir = "country211"

    def __init__(self, root, num_shots=0, seed=1, subsample_classes='all'):
        root = os.path.abspath(os.path.expanduser(root))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        # Prefer explicit split directories if they exist
        train_dir = os.path.join(self.dataset_dir, "train")
        valid_dir = os.path.join(self.dataset_dir, "valid")
        test_dir = os.path.join(self.dataset_dir, "test")

        use_folder_splits = os.path.isdir(train_dir) and os.path.isdir(valid_dir) and os.path.isdir(test_dir)

        if use_folder_splits:
            # Build ISO2 -> name mapping (allow override via json)
            iso_map = self._load_iso2_to_name()
            # Create unified code set across splits to keep label mapping stable
            all_codes = set()
            for d in [train_dir, valid_dir, test_dir]:
                if os.path.isdir(d):
                    for code in listdir_nohidden(d):
                        code_path = os.path.join(d, code)
                        if os.path.isdir(code_path):
                            all_codes.add(code)
            # Validate code format and mapping coverage early
            override_path = os.path.join(self.dataset_dir, 'iso2_to_name.json')
            pat = re.compile(r'^[A-Z]{2}$')
            invalid_codes = sorted([c for c in all_codes if not (pat.match(c) or c == 'XK')])
            if invalid_codes:
                raise ValueError(
                    "Country211: Found invalid ISO2 code folder names: {}. "
                    "Codes must be two uppercase letters (e.g., 'US', 'CN') or 'XK'. "
                    "Please rename these folders accordingly.".format(invalid_codes)
                )
            unknown_codes = sorted([c for c in all_codes if c not in iso_map])
            if unknown_codes:
                raise ValueError(
                    "Country211: ISO2 codes missing from mapping: {}. "
                    "Add them to {} as a JSON dict, e.g., {\"XX\": \"Country Name\"}.".format(
                        unknown_codes, override_path
                    )
                )
            # Sort by human name (fallback to code) for stable labels
            def _code_to_name(c):
                return iso_map.get(c, c)
            sorted_codes = sorted(list(all_codes), key=lambda c: _code_to_name(c))
            code_to_label = {c: i for i, c in enumerate(sorted_codes)}

            train = self._read_split_dir(train_dir, code_to_label, iso_map)
            val = self._read_split_dir(valid_dir, code_to_label, iso_map)
            test = self._read_split_dir(test_dir, code_to_label, iso_map)
        else:
            # Fallbacks: JSON split or naive folder split
            image_dir = os.path.join(self.dataset_dir, "images")
            self.image_dir = image_dir if os.path.isdir(image_dir) else self.dataset_dir
            split_path = os.path.join(self.dataset_dir, "split_zhou_Country211.json")
            if os.path.exists(split_path):
                train, val, test = OxfordPets.read_split(split_path, self.image_dir)
            else:
                train, val, test = DTD.read_and_split_data(self.image_dir)
                OxfordPets.save_split(train, val, test, split_path, self.image_dir)

        if num_shots >= 1:
            preprocessed = os.path.join(self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl")
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        subsample = subsample_classes
        train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)

        # Templates provided by user
        self.templates = [
            lambda c: f'a photo i took in {c}.',
            lambda c: f'a photo i took while visiting {c}.',
            lambda c: f'a photo from my home country of {c}.',
            lambda c: f'a photo from my visit to {c}.',
            lambda c: f'a photo showing the country of {c}.',
        ]

        super().__init__(train_x=train, val=val, test=test)

    def _read_split_dir(self, split_dir, code_to_label, iso_map):
        items = []
        if not os.path.isdir(split_dir):
            return items
        codes = listdir_nohidden(split_dir)
        for code in codes:
            class_dir = os.path.join(split_dir, code)
            if not os.path.isdir(class_dir):
                continue
            label = code_to_label.get(code)
            if label is None:
                raise ValueError(
                    f"Country211: Inconsistent label mapping for code '{code}' in split '{split_dir}'. "
                    f"This indicates an internal mismatch between discovered codes and label map."
                )
            try:
                cname = iso_map[code]
            except KeyError:
                raise ValueError(
                    f"Country211: Missing ISO mapping for code '{code}'. "
                    f"Please add it to '{os.path.join(self.dataset_dir, 'iso2_to_name.json')}'."
                )
            for fname in listdir_nohidden(class_dir):
                impath = os.path.join(class_dir, fname)
                items.append(Datum(impath=impath, label=label, classname=cname))
        return items

    def _load_iso2_to_name(self):
        """Load ISO2->country name mapping, allow local JSON override.
        Fallback to built-in mapping; unknown codes map to themselves at usage time.
        """
        override_path = os.path.join(self.dataset_dir, 'iso2_to_name.json')
        if os.path.exists(override_path):
            try:
                data = read_json(override_path)
                if not isinstance(data, dict):
                    raise ValueError(
                        f"Country211: Expected a JSON object in '{override_path}', got {type(data)}"
                    )
                # Validate keys as ISO2 (two uppercase letters) or 'XK', and values are non-empty strings
                pat = re.compile(r'^[A-Z]{2}$')
                bad_keys = [k for k in data.keys() if not (isinstance(k, str) and (pat.match(k) or k == 'XK'))]
                bad_vals = [k for k, v in data.items() if not (isinstance(v, str) and v.strip())]
                if bad_keys or bad_vals:
                    raise ValueError(
                        (
                            "Country211: Invalid iso2_to_name.json entries. "
                            f"Bad keys (must be ISO2 like 'US'): {bad_keys}. "
                            f"Bad values (must be non-empty strings) for keys: {bad_vals}."
                        )
                    )
                return data
            except Exception as e:
                raise ValueError(f"Country211: Failed to load '{override_path}': {e}")
        # Built-in mapping aligned with provided classes
        return {
            'AD': 'Andorra',
            'AE': 'United Arab Emirates',
            'AF': 'Afghanistan',
            'AG': 'Antigua and Barbuda',
            'AI': 'Anguilla',
            'AL': 'Albania',
            'AM': 'Armenia',
            'AO': 'Angola',
            'AQ': 'Antarctica',
            'AR': 'Argentina',
            'AT': 'Austria',
            'AU': 'Australia',
            'AW': 'Aruba',
            'AX': 'Aland Islands',
            'AZ': 'Azerbaijan',
            'BA': 'Bosnia and Herzegovina',
            'BB': 'Barbados',
            'BD': 'Bangladesh',
            'BE': 'Belgium',
            'BF': 'Burkina Faso',
            'BG': 'Bulgaria',
            'BH': 'Bahrain',
            'BJ': 'Benin',
            'BM': 'Bermuda',
            'BN': 'Brunei Darussalam',
            'BO': 'Bolivia',
            'BQ': 'Bonaire, Saint Eustatius and Saba',
            'BR': 'Brazil',
            'BS': 'Bahamas',
            'BT': 'Bhutan',
            'BW': 'Botswana',
            'BY': 'Belarus',
            'BZ': 'Belize',
            'CA': 'Canada',
            'CD': 'DR Congo',
            'CF': 'Central African Republic',
            'CH': 'Switzerland',
            'CI': "Cote d'Ivoire",
            'CK': 'Cook Islands',
            'CL': 'Chile',
            'CM': 'Cameroon',
            'CN': 'China',
            'CO': 'Colombia',
            'CR': 'Costa Rica',
            'CU': 'Cuba',
            'CV': 'Cabo Verde',
            'CW': 'Curacao',
            'CY': 'Cyprus',
            'CZ': 'Czech Republic',
            'DE': 'Germany',
            'DK': 'Denmark',
            'DM': 'Dominica',
            'DO': 'Dominican Republic',
            'DZ': 'Algeria',
            'EC': 'Ecuador',
            'EE': 'Estonia',
            'EG': 'Egypt',
            'ES': 'Spain',
            'ET': 'Ethiopia',
            'FI': 'Finland',
            'FJ': 'Fiji',
            'FK': 'Falkland Islands',
            'FO': 'Faeroe Islands',
            'FR': 'France',
            'GA': 'Gabon',
            'GB': 'United Kingdom',
            'GD': 'Grenada',
            'GE': 'Georgia',
            'GF': 'French Guiana',
            'GG': 'Guernsey',
            'GH': 'Ghana',
            'GI': 'Gibraltar',
            'GL': 'Greenland',
            'GM': 'Gambia',
            'GP': 'Guadeloupe',
            'GR': 'Greece',
            'GS': 'South Georgia and South Sandwich Is.',
            'GT': 'Guatemala',
            'GU': 'Guam',
            'GY': 'Guyana',
            'HK': 'Hong Kong',
            'HN': 'Honduras',
            'HR': 'Croatia',
            'HT': 'Haiti',
            'HU': 'Hungary',
            'ID': 'Indonesia',
            'IE': 'Ireland',
            'IL': 'Israel',
            'IM': 'Isle of Man',
            'IN': 'India',
            'IQ': 'Iraq',
            'IR': 'Iran',
            'IS': 'Iceland',
            'IT': 'Italy',
            'JE': 'Jersey',
            'JM': 'Jamaica',
            'JO': 'Jordan',
            'JP': 'Japan',
            'KE': 'Kenya',
            'KG': 'Kyrgyz Republic',
            'KH': 'Cambodia',
            'KN': 'St. Kitts and Nevis',
            'KP': 'North Korea',
            'KR': 'South Korea',
            'KW': 'Kuwait',
            'KY': 'Cayman Islands',
            'KZ': 'Kazakhstan',
            'LA': 'Laos',
            'LB': 'Lebanon',
            'LC': 'St. Lucia',
            'LI': 'Liechtenstein',
            'LK': 'Sri Lanka',
            'LR': 'Liberia',
            'LT': 'Lithuania',
            'LU': 'Luxembourg',
            'LV': 'Latvia',
            'LY': 'Libya',
            'MA': 'Morocco',
            'MC': 'Monaco',
            'MD': 'Moldova',
            'ME': 'Montenegro',
            'MF': 'Saint-Martin',
            'MG': 'Madagascar',
            'MK': 'Macedonia',
            'ML': 'Mali',
            'MM': 'Myanmar',
            'MN': 'Mongolia',
            'MO': 'Macau',
            'MQ': 'Martinique',
            'MR': 'Mauritania',
            'MT': 'Malta',
            'MU': 'Mauritius',
            'MV': 'Maldives',
            'MW': 'Malawi',
            'MX': 'Mexico',
            'MY': 'Malaysia',
            'MZ': 'Mozambique',
            'NA': 'Namibia',
            'NC': 'New Caledonia',
            'NG': 'Nigeria',
            'NI': 'Nicaragua',
            'NL': 'Netherlands',
            'NO': 'Norway',
            'NP': 'Nepal',
            'NZ': 'New Zealand',
            'OM': 'Oman',
            'PA': 'Panama',
            'PE': 'Peru',
            'PF': 'French Polynesia',
            'PG': 'Papua New Guinea',
            'PH': 'Philippines',
            'PK': 'Pakistan',
            'PL': 'Poland',
            'PR': 'Puerto Rico',
            'PS': 'Palestine',
            'PT': 'Portugal',
            'PW': 'Palau',
            'PY': 'Paraguay',
            'QA': 'Qatar',
            'RE': 'Reunion',
            'RO': 'Romania',
            'RS': 'Serbia',
            'RU': 'Russia',
            'RW': 'Rwanda',
            'SA': 'Saudi Arabia',
            'SB': 'Solomon Islands',
            'SC': 'Seychelles',
            'SD': 'Sudan',
            'SE': 'Sweden',
            'SG': 'Singapore',
            'SH': 'St. Helena',
            'SI': 'Slovenia',
            'SJ': 'Svalbard and Jan Mayen Islands',
            'SK': 'Slovakia',
            'SL': 'Sierra Leone',
            'SM': 'San Marino',
            'SN': 'Senegal',
            'SO': 'Somalia',
            'SS': 'South Sudan',
            'SV': 'El Salvador',
            'SX': 'Sint Maarten',
            'SY': 'Syria',
            'SZ': 'Eswatini',
            'TG': 'Togo',
            'TH': 'Thailand',
            'TJ': 'Tajikistan',
            'TL': 'Timor-Leste',
            'TM': 'Turkmenistan',
            'TN': 'Tunisia',
            'TO': 'Tonga',
            'TR': 'Turkey',
            'TT': 'Trinidad and Tobago',
            'TW': 'Taiwan',
            'TZ': 'Tanzania',
            'UA': 'Ukraine',
            'UG': 'Uganda',
            'US': 'United States',
            'UY': 'Uruguay',
            'UZ': 'Uzbekistan',
            'VA': 'Vatican',
            'VE': 'Venezuela',
            'VG': 'British Virgin Islands',
            'VI': 'United States Virgin Islands',
            'VN': 'Vietnam',
            'VU': 'Vanuatu',
            'WS': 'Samoa',
            'XK': 'Kosovo',
            'YE': 'Yemen',
            'ZA': 'South Africa',
            'ZM': 'Zambia',
            'ZW': 'Zimbabwe',
        }
