(() => {
  const sectionIds = ["overview", "demo", "l3rocc", "occplanner", "results"];
  const sections = sectionIds
    .map((id) => document.getElementById(id))
    .filter(Boolean);
  const navLinks = [...document.querySelectorAll(
    'nav a[href^="#"], .mobile-nav-panel a[href^="#"]'
  )];
  const mobileNav = document.querySelector(".mobile-nav");

  const setActiveSection = () => {
    const marker = window.scrollY + window.innerHeight * 0.3;
    let activeId = "";

    for (const section of sections) {
      if (section.offsetTop <= marker) activeId = section.id;
    }

    for (const link of navLinks) {
      const isActive = link.getAttribute("href") === "#" + activeId;
      if (isActive) {
        link.setAttribute("aria-current", "location");
      } else {
        link.removeAttribute("aria-current");
      }
    }
  };

  let ticking = false;
  const requestUpdate = () => {
    if (ticking) return;
    ticking = true;
    window.requestAnimationFrame(() => {
      setActiveSection();
      ticking = false;
    });
  };

  for (const link of navLinks) {
    link.addEventListener("click", () => {
      if (mobileNav) mobileNav.removeAttribute("open");
    });
  }

  document.addEventListener("click", (event) => {
    if (mobileNav?.open && !mobileNav.contains(event.target)) {
      mobileNav.removeAttribute("open");
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && mobileNav?.open) {
      mobileNav.removeAttribute("open");
      mobileNav.querySelector("summary")?.focus();
    }
  });

  window.addEventListener("scroll", requestUpdate, { passive: true });
  window.addEventListener("resize", requestUpdate);
  setActiveSection();
})();
