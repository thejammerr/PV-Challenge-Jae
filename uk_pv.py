# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import pandas as pd
import xarray as xr
import datasets
from datetime import timedelta


# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@InProceedings{uk_pv,
title = {UK PV solar generation dataset},
author={Open Climate Fix.
},
year={2022}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
# UK PV dataset
PV solar generation data from the UK. 
This dataset contains dataa from 1311 PV systems from 2018-01-01 to 2021-10-27. 
The time series of solar generation is in 5 minutes chunks. 
This data is from collected from live PV systems in the UK. We have obfuscated the location of the pv systems for privacy. 
If you are the owner of a PV system in the dataset, and do not want this data to be shared, 
please do get in contact with info@openclimatefix.org.
## Files
The dataset contains two files
- metadata.csv: Data about the PV systems, e.g location
- pv.netcdf: Time series of PV solar generation
### metadata.csv
Metadata of the different PV systems. 
Note that there are extra PV systems in this metadata that do not appear in the pv timeseries data
The csv columns are
- ss_id: the id of the system
- latitude_rounded: latitude of the pv system, but rounded to approximately the nearest km
- longitude_rounded: latitude of the pv system, but rounded to approximately the nearest km
- llsoacd: TODO
- orientation: The orientation of the pv system
- tilt: The tilt of the pv system
- kwp: The capacity of the pv system
- operational_at: the datetime the pv system started working
### pv.netcdf
Time series data of pv solar generation data is in a [xarray](https://docs.xarray.dev/en/stable/) format.
The data variables are the same as 'ss_id' in the metadata. 
Each data variable contains the solar generation (in kw) for that pv system. 
The ss_id's here are a subset of the all the ss_id's in the metadata 
The co-ordinates of the date are 'datetime' which is the datetime of the solar generation reading.
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = "CC-BY-4.0 - https://creativecommons.org/licenses/by/4.0/"

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLS = {
    'metadata': 'metadata.csv',
    'data': 'pv.netcdf'
}


# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class UKPV(datasets.GeneratorBasedBuilder):
    """ PV solar generation data from the UK. """

    VERSION = datasets.Version("0.0.1")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="uk_pv", version=VERSION, description="PV solar generation data from the UK."),
        datasets.BuilderConfig(name="uk_pv_sat", version=VERSION, description="PV solar generation data from the UK, along with EUMETSAT HRV satellite data."),
    ]

    DEFAULT_CONFIG_NAME = "uk_pv"  # It's not mandatory to have a default configuration. Just use one if it make sense.

    def _info(self):
        if self.config.name == "uk_pv":
            features = datasets.Features(
                {
                    "timestamp": datasets.Value("time64[ns]"),
                    "pv_yield": datasets.Array2D(shape=(288,1311), dtype="float64")
                    # These are the features of your dataset like images, labels ...
                }
            )
        else: # Also include satellite imagery
            features = datasets.Features(
                {
                    "timestamp": datasets.Value("time64[ns]"),
                    "pv_yield": datasets.Array2D(shape=(288,1311), dtype="float64"),
                    "sat_image": datasets.Array3D(shape=(288,891,1843), dtype="float64"),
                    "x_coordinates": datasets.Array2D(shape=(891,1843), dtype="float64"),
                    "y_coordinates": datasets.Array2D(shape=(891,1843), dtype="float64")
                    # These are the features of your dataset like images, labels ...
                }
            )
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):

        data_dir = dl_manager.download(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir['data'],
                    "time_range": slice("2018-01-01", "2020-12-31"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "filepath": data_dir['data'],
                    "time_range": slice("2021-01-01", "2021-12-31"),
                    "split": "test"
                },
            ),
        ]

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, filepath, time_range, split):

        # load the timeseries data
        if self.config.name == "uk_pv":
            with xr.open_dataset(filepath, engine="h5netcdf") as pv_power:

                pv_power = pv_power.sel(datetime=time_range)
                pv_power_df = pv_power.to_dataframe()

                # get unqiue dates
                dates = pd.Series(pd.to_datetime(pv_power_df.index).date).unique()

                for key, timestamp in enumerate(dates):

                    # just select one day of data
                    timestamp = pd.to_datetime(timestamp)
                    entry = pv_power_df[(pv_power_df.index >= timestamp) & (pv_power_df.index < timestamp+timedelta(days=1))]

                    yield key, {'timestamp':timestamp,
                                'pv_yield':entry}
        else:
            sat_data = xr.open_zarr("gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v3/eumetsat_seviri_hrv_uk.zarr", chunks="auto")

            with xr.open_dataset(filepath, engine="h5netcdf") as pv_power:

                pv_power = pv_power.sel(datetime=time_range)
                pv_power_df = pv_power.to_dataframe()

                # get unqiue dates
                dates = pd.Series(pd.to_datetime(pv_power_df.index).date).unique()

                for key, timestamp in enumerate(dates):
                    # Check timestamp range exists in satellite data


                    # just select one day of data
                    timestamp = pd.to_datetime(timestamp)
                    entry = pv_power_df[(pv_power_df.index >= timestamp) & (pv_power_df.index < timestamp+timedelta(days=1))]
                    sat_images = sat_data.sel(time=slice(timestamp,timestamp+timedelta(days=1)))
                    if len(sat_images.time.values) == 288: # Full set of images
                        yield key, {'timestamp':timestamp,
                                    'pv_yield':entry,
                                    "x_coordinates": sat_images.x_osgb.values,
                                    "y_coordinates": sat_images.y_osgb.values,
                                    "sat_image": sat_images.values}

