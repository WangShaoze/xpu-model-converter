# -*- coding: utf-8 -*-
"""gunicorn 启动器(公共 Runtime)。

启动方式与客户现场部署脚本一致: ``python runtime_gunicorn.py``。
单机多卡场景下通过 ``on_starting`` / ``pre_fork`` / ``child_exit`` 钩子把
XPU 卡分配给各 worker 进程。
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def build_application(app_uri, options):
    """构造 gunicorn Application(延迟导入, 便于无 gunicorn 环境做静态检查)。"""
    from gunicorn.app.base import BaseApplication
    from gunicorn import util

    class NwaiGunicornApplication(BaseApplication):
        """自定义 Application: 直接使用 settings 中的 gunicorn 选项。"""

        def __init__(self, uri, opts):
            self.app_uri = uri
            self.options = dict(opts or {})
            super(NwaiGunicornApplication, self).__init__()

        def load_config(self):
            for key, value in self.options.items():
                if value is None:
                    continue
                if key in self.cfg.settings:
                    self.cfg.set(key.lower(), value)

        def load(self):
            return util.import_app(self.app_uri)

    return NwaiGunicornApplication(app_uri, options)


def main():
    import runtime_settings

    options = runtime_settings.get_gunicorn_options()
    app_uri = options.pop("app_uri", "runtime_server:app")
    application = build_application(app_uri, options)
    application.run()


if __name__ == "__main__":
    main()
