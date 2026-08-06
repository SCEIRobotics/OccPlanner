(() => {
  const sectionIds = ["real-demos", "demo", "overview", "occplanner", "l3rocc", "results"];
  const sections = sectionIds
    .map((id) => document.getElementById(id))
    .filter(Boolean);
  const navLinks = [...document.querySelectorAll(
    'nav a[href^="#"], .mobile-nav-panel a[href^="#"]'
  )];
  const mobileNav = document.querySelector(".mobile-nav");
  const autoplayVideos = [...document.querySelectorAll("video.viewport-autoplay")];

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

  if ("IntersectionObserver" in window) {
    const videoObserver = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        const video = entry.target;
        const shouldPlay = entry.isIntersecting && entry.intersectionRatio >= 0.35;
        video.dataset.inViewport = shouldPlay ? "true" : "false";
        if (shouldPlay && !document.hidden) {
          video.play().catch(() => {});
        } else {
          video.pause();
        }
      }
    }, { threshold: [0, 0.35, 0.75] });

    for (const video of autoplayVideos) {
      video.muted = true;
      videoObserver.observe(video);
    }

    document.addEventListener("visibilitychange", () => {
      for (const video of autoplayVideos) {
        if (document.hidden) {
          video.pause();
        } else if (video.dataset.inViewport === "true") {
          video.play().catch(() => {});
        }
      }
    });
  } else {
    for (const video of autoplayVideos) {
      video.muted = true;
      video.play().catch(() => {});
    }
  }

  setActiveSection();
})();
