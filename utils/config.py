import yaml
import os

class Config():
    def __init__(self, config_file_pth):
        self.configurations = self.load_config_file(config_file_pth)
    def load_config_file(self, config_pth):
        with open(config_pth) as config:
            configurations = yaml.full_load(config)

        if config is None:
            print("No configurations were provided. Exiting...")
            exit(0)

        return configurations

    def get_config_setting(self, setting_name):
        return self.configurations.get(setting_name, None)
