"use strict";
window.I18n = (() => {
  const supported = ["en", "fr", "es", "zh"];
  const tissues = { "乳腺": "tissue.breast", "皮肤": "tissue.skin", "脑皮层": "tissue.cortex" };
  let language = "en", messages = {};
  try {
    const saved = localStorage.getItem("dragh2st.language");
    if (supported.includes(saved)) language = saved;
  } catch (_) { /* Storage is optional. */ }
  function t(key, values = {}) {
    const template = messages[language]?.[key] ?? messages.en?.[key] ?? key;
    return template.replace(/\{(\w+)\}/g, (match, name) => String(values[name] ?? match));
  }
  function number(value, digits) {
    const options = digits === undefined ? { maximumFractionDigits: 0 } : {
      minimumFractionDigits: digits, maximumFractionDigits: digits,
    };
    return new Intl.NumberFormat(language, options).format(value);
  }
  function apply() {
    document.documentElement.lang = language === "zh" ? "zh-CN" : language;
    document.querySelectorAll("[data-i18n]").forEach((node) => {
      node.textContent = t(node.dataset.i18n);
    });
    for (const attribute of ["title", "placeholder", "aria-label"]) {
      document.querySelectorAll(`[data-i18n-${attribute}]`).forEach((node) => {
        node.setAttribute(attribute, t(node.getAttribute(`data-i18n-${attribute}`)));
      });
    }
    document.getElementById("language-select").value = language;
  }
  function setLanguage(next) {
    language = supported.includes(next) ? next : "en";
    try { localStorage.setItem("dragh2st.language", language); } catch (_) {}
    apply();
  }
  function tissueName(value) { return tissues[value] ? t(tissues[value]) : value; }
  function tissueValue(value) {
    const normalized = value.trim().toLocaleLowerCase();
    for (const [canonical, key] of Object.entries(tissues)) {
      if (canonical === value || supported.some((lang) => messages[lang]?.[key]?.toLocaleLowerCase() === normalized)) return canonical;
    }
    return value.trim();
  }
  const ready = fetch("/translations.json").then((response) => {
    if (!response.ok) throw new Error("Unable to load translations.");
    return response.json();
  }).then((result) => { messages = result; apply(); });
  return { t, number, tissueName, tissueValue, apply, setLanguage, ready,
    get language() { return language; } };
})();
